require 'torch'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'
require 'optim'
require 'pl'
require 'paths'
require 'image'
util = require 'utils.base'

----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  --lrD               (default 0.0002)            learning rate
  --lrG               (default 0.0002)            learning rate
  --learningRateDecay (default 0)
  --beta1            (default 0.5)               momentum term for adam
  -b,--batchSize     (default 100)               batch size
  -g,--gpu           (default 0)                 gpu to use
  -s,--save          (default "logs/")           base directory to save logs
  --optimizer        (default "adam")            "adam" | "sgd" | "adagrad"
  --nEpochs          (default 100)              max training epochs
  --seed             (default 1)                 random seed
  --imageSize        (default 64)                image resolution (64 or 96)
  --backend          (default "cudnn")            "cunn" | "cudnn"
  --epochSize        (default 100000)             number of samples per epoch
  --modelD            (default "vgg")             "dcgan" | "vgg"
  --modelG            (default "dcgan")             "dcgan" | "unet"
  --lrC              (default 1)                 multiplier on gradient from netC to netD for real data
  --patchMin         (default 32)                min size of hole to cut
  --patchMax         (default 32)                max size of hole to cut
  --nPatch           (default 1)                 # of patches to cut out
  --curriculum                                   if true increase patch size based on curriculum
  --dataset          (default "stl")            "stl" | "cifar"
  --dataPath         (default "")               path to dataset
  --noiseDim         (default 100)              dim of noise vector when model is stochastic
  --stochastic                                  make generation stochastic
  --classifyFake                                if true then pass augmented data through classifier
  --classAdversary                              if true get generator to fool classifier
  --nlabelled        (default 4000)             for cifar10
  --fold             (default 1)                pre-defined fold for STL-10
  --reverse                                     if set predict context given hole
]]

os.execute('mkdir -p ' .. opt.save .. '/gen/')

assert(optim[opt.optimizer] ~= nil, 'unknown optimizer: ' .. opt.optimizer)
opt.optimizer = optim[opt.optimizer]

print(opt)
write_opt(opt)

-- setup some stuff
torch.setnumthreads(4)
print('<torch> set nb of threads to ' .. torch.getnumthreads())
torch.setdefaulttensortype('torch.FloatTensor')

cutorch.setDevice(opt.gpu + 1)
print('<gpu> using device ' .. opt.gpu)

torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)
math.randomseed(opt.seed)

opt.geometry = {3, opt.imageSize, opt.imageSize}

paths.dofile(('models/%s_%s_%d.lua'):format(opt.modelG, opt.modelD,  opt.imageSize))

local criterionD = nn.BCECriterion()
local criterionC = nn.ClassNLLCriterion()
local confusion = optim.ConfusionMatrix(10)

netG:cuda()
netD:cuda()
netC:cuda()
criterionC:cuda()
criterionD:cuda()

params_G, grads_G = netG:getParameters()
params_D, grads_D = netD:getParameters()
params_C, grads_C = netC:getParameters()

local target_real = 0.9 
local target_fake = 0
local label = torch.CudaTensor(opt.batchSize)
local class_targets = torch.CudaTensor(opt.batchSize)
local x = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
local gen = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
local mask = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
local reverse_mask = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
local noise = torch.CudaTensor(opt.batchSize, opt.noiseDim, 1, 1)
local d1, d2 = unpack(netD:forward({ {x:zero(), mask:zero()}, {x:zero(), reverse_mask:zero()} }))
local zeros1 = torch.CudaTensor(d1:size()):fill(0)
local zeros2 = torch.CudaTensor(d2:size()):fill(0)
local rp = torch.LongTensor(opt.batchSize)


optimStateG = {
   learningRate = opt.lrG,
   learningRateDecay = opt.learningRateDecay,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lrD,
   learningRateDecay = opt.learningRateDecay,
   beta1 = opt.beta1,
}
optimStateC = {
   learningRate = opt.lrD,
   learningRateDecay = opt.learningRateDecay,
   beta1 = opt.beta1,
}

trainLoggerC = optim.Logger(opt.save .. '/trainC.log')
testLoggerC = optim.Logger(opt.save .. '/testC.log')
trainLoggerD = optim.Logger(opt.save .. '/trainD.log')
testLoggerD = optim.Logger(opt.save .. '/testD.log')

opt.translate = true
opt.scale = true 
require 'data.data'

trainData:plotData(opt.save .. '/data.png')

local function sampleNoise(z)
  z:normal()
end

function sampleMask(mask, reverse_mask)
  local mask_val, reverse_val
  if opt.reverse then
    mask_val = 0
    reverse_val = 1
  else
    mask_val = 1
    reverse_val = 0
  end
  mask:fill(mask_val)
  reverse_mask:fill(reverse_val)
  for n = 1,mask:size(1) do
    for p = 1,math.random(opt.nPatch) do
      local patch_x = math.random(opt.patchMin, opt.patchMax)
      local patch_y = math.random(opt.patchMin, opt.patchMax)
      local sx = math.random(1, mask[n]:size(2) - patch_x)
      local sy = math.random(1, mask[n]:size(3) - patch_y)
      mask[n][{ {}, {sx,sx+patch_x-1}, {sy,sy+patch_y-1} }]:fill(reverse_val)
      reverse_mask[n][{ {}, {sx,sx+patch_x-1}, {sy,sy+patch_y-1} }]:fill(mask_val)
    end
  end
end

function trainC(dataset)
  grads_D:zero()
  grads_C:zero()
  local top1_real, top1_fake = 0, 0

  -- real data
  dataset:getBatch(x, class_targets, nil, opt.batchSize)
  sampleMask(mask, reverse_mask)
  local _, latent = unpack(netD:forward({ {x, mask}, {x, reverse_mask} }))
  local out_c = netC:forward(latent)
  local err_c = criterionC:forward(out_c, class_targets)
  local dout_c = criterionC:backward(out_c, class_targets)
  local dlatent = netC:backward(latent, dout_c)
  dlatent:mul(opt.lrC)
  netD:backward({ {x, mask}, {x, reverse_mask} }, {zeros1, dlatent})
  top1_real = classResults(out_c, class_targets)

  if opt.classifyFake then
    -- fake data
    dataset:getBatch(x, class_targets, nil, opt.batchSize)
    sampleMask(mask, reverse_mask)
    sampleNoise(noise)
    gen:copy(netG:forward({ {x, mask}, noise }))
    local _, latent = unpack(netD:forward({ {x, mask}, {gen, reverse_mask} }))
    local out_c = netC:forward(latent)
    local err_c = criterionC:forward(out_c, class_targets)
    local dout_c = criterionC:backward(out_c, class_targets)
    local dlatent = netC:backward(latent, dout_c)
    dlatent:mul(opt.lrC)
    netD:backward({ {x, mask}, {gen, reverse_mask} }, {zeros1, dlatent})
    top1_fake = classResults(out_c, class_targets)
  end

  opt.optimizer(function() return 0, grads_D end, params_D, optimStateD)
  opt.optimizer(function() return 0, grads_C end, params_C, optimStateC)

  return top1_real, top1_fake
end

function trainG(dataset)
  for _, m in pairs(netG:listModules()) do zeroBias(m) end
  for _, m in pairs(netD:listModules()) do zeroBias(m) end
  grads_D:zero()

  -- real
  label:fill(target_real)
  dataset:getBatch(x, class_targets)
  sampleMask(mask, reverse_mask)
  local out_d, _ = unpack(netD:forward({ {x, mask}, {x, reverse_mask} }))
  local errR = criterionD:forward(out_d, label)
  local dout_d = criterionD:backward(out_d, label)
  netD:backward({ {x, mask}, {x, reverse_mask} }, {dout_d, zeros2})

  -- fake
  label:fill(target_fake)
  dataset:getBatch(x, class_targets)
  sampleMask(mask, reverse_mask)
  sampleNoise(noise)
  gen:copy(netG:forward({ {x, mask}, noise }))
  local out_d, _ = unpack(netD:forward({ {x, mask}, {gen, reverse_mask} }))
  local errF = criterionD:forward(out_d, label)
  local dout_d = criterionD:backward(out_d, label)
  netD:backward({ {x, mask}, {gen, reverse_mask} }, {dout_d, zeros2})

  opt.optimizer(function() return 0, grads_D end, params_D, optimStateD)

  for _, m in pairs(netG:listModules()) do zeroBias(m) end
  for _, m in pairs(netD:listModules()) do zeroBias(m) end
  grads_G:zero()

  -- train encoder/decoder
  label:fill(target_real)
  netD:forward({ {x, mask}, {gen, reverse_mask} })
  criterionD:forward(out_d, label)
  local dout_d = criterionD:backward(out_d, label)
  local dgen = netD:updateGradInput({ {x, mask}, {gen, reverse_mask} }, {dout_d, zeros2})[2][1]
  local dgen = netD:backward({ {x, mask}, {gen, reverse_mask} }, {dout_d, zeros2})[2][1]
  netG:backward({ {x, mask}, noise }, dgen)

  if opt.classAdversary then
    dataset:getBatch(x, class_targets, nil, opt.batchSize)
    sampleMask(mask, reverse_mask)
    sampleNoise(noise)
    gen:copy(netG:forward({ {x, mask}, noise }))
    local out_d, latent = unpack(netD:forward({ {x, mask}, {gen, reverse_mask} }))
    local out_c = netC:forward(latent)
    criterionD:forward(out_d, label)
    criterionC:forward(out_c, class_targets)
    local dout_d = criterionD:backward(out_d, label)
    local dout_c = criterionC:backward(out_c, class_targets)
    local dlatent = netC:backward(latent, dout_c)
    local dgen = netD:backward({ {x, mask}, {gen, reverse_mask} }, {dout_d, latent})[2][1]
    netG:backward({ {x, mask}, noise }, dgen)
  end
  opt.optimizer(function() return 0, grads_G end, params_G, optimStateG)

  opt.optimizer(function() return 0, grads_G end, params_G, optimStateG)
  return errR, errF
end

function test(dataset, idx)
  local top1_real, top1_fake = 0, 0
  -- real
  label:fill(target_real)
  dataset:getBatch(x, class_targets)
  sampleMask(mask, reverse_mask)
  local out_d, latent = unpack(netD:forward({ {x, mask}, {x, reverse_mask} }))
  local errR = criterionD:forward(out_d, label)
  local out_c = netC:forward(latent)
  top1_real = classResults(out_c, class_targets)

  -- fake data
  label:fill(target_fake)
  dataset:getBatch(x, class_targets)
  sampleMask(mask, reverse_mask)
  sampleNoise(noise)
  gen:copy(netG:forward({ {x, mask}, noise }))
  local out_d, latent = unpack(netD:forward({ {x, mask}, {gen, reverse_mask} }))
  local errF = criterionD:forward(out_d, label)
  local out_c = netC:forward(latent)
  top1_fake = classResults(out_c, class_targets)

  return errR, errF, top1_real, top1_fake
end

local function plot(dataset, fname, N)
  dataset:getBatch(x)
  sampleMask(mask, reverse_mask)
  sampleNoise(noise)
  util.plot(x, mask, reverse_mask, noise, fname, N, epoch, opt)
end

finalMin = opt.patchMin
finalMax= opt.patchMax --final max patch size for curriculum
if opt.curriculum then
  if not opt.reverse and opt.stochastic then
    opt.patchMin = opt.patchMax
  else
    opt.patchMax = opt.PatchMin
  end
end
best = 0
while true do
  collectgarbage()
  epoch = epoch or 1
  print('\n<trainer> Epoch ' .. epoch)
  local errR, errF, top1_real, top1_fake = 0, 0, 0, 0
  local nTrain = opt.epochSize
  local counter = 0
  for i = 1,nTrain, opt.batchSize do
    xlua.progress(i, nTrain)
    local err, erf = trainG(trainData)
    local rt1, ft1 = trainC(trainData)
    errR = errR + err
    errF = errF + erf
    top1_real = top1_real + rt1
    top1_fake = top1_fake + ft1
    counter = counter + opt.batchSize
  end
  errR = errR / counter 
  errF = errF / counter
  local train_acc = 100*top1_real/counter
  print('errR = ' .. errR)
  print('errF = ' .. errF)
  print('real top1 = ' .. train_acc)
  if opt.classifyFake then print('fake top1 = ' .. 100*top1_fake/counter) end
  trainLoggerC:add{train_acc}
  trainLoggerD:add{(errR+errF)/2}

  print('\n<tester> Epoch ' .. epoch)
  local errR, errF, top1_real, top1_fake = 0, 0, 0, 0
  local nTest = valData:size()
  local counter = 0
  for i = 1,nTest, opt.batchSize do
    xlua.progress(i, nTest)
    local err, erf, rt1, ft1 = test(valData)
    errR = errR + err
    errF = errF + erf
    top1_real = top1_real + rt1
    top1_fake = top1_fake + ft1
    counter = counter + opt.batchSize
  end
  errR = errR / counter
  errF = errF / counter
  local test_acc = 100*top1_real/counter
  print('errR = ' .. errR)
  print('errF = ' .. errF)
  print('real top1 = ' .. test_acc)
  if opt.classifyFake then print('fake top1 = ' .. 100*top1_fake/counter) end
  testLoggerC:add{test_acc}
  testLoggerD:add{(errR+errF)/2}

  plot(valData, 'val')
  util.plotAcc(trainLoggerC.symbols[1], testLoggerC.symbols[1], opt.save .. '/accC', 'Classifier Accuracy (epoch ' .. epoch .. ')')
  util.plotAcc(trainLoggerD.symbols[1], testLoggerD.symbols[1], opt.save .. '/errD', 'Discriminator error (epoch ' .. epoch .. ')')
  util.bistro_log({epoch = epoch, train_acc = train_acc, test_acc = test_acc, errR = errR, errF = errF}, opt)

  epoch = epoch + 1

  if epoch % 1 == 0 then
    print('Saving model: ' .. opt.save .. '/model.t7')
    torch.save(opt.save .. '/model.t7', {netD=util.sanitize(netD), netG=util.sanitize(netG), netC=util.sanitize(netC)})
  end

  if opt.lrC > 0 and test_acc > best then
    print('New best accuracy: ' .. test_acc)
    print('Saving model: ' .. opt.save .. '/model_best.t7')
    torch.save(opt.save .. '/model_best.t7', {netD=util.sanitize(netD), netC=util.sanitize(netC)})
    best = test_acc
  end

  if opt.curriculum and epoch % 5 == 0 then
    if not opt.reverse and opt.stochastic then
      opt.patchMin = math.max(opt.patchMin - 8, finalMin)
    else
      opt.patchMax = math.min(opt.patchMax + 8, finalMax)
    end
  end
  if epoch > opt.nEpochs then break end
end
