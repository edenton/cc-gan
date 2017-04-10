-- some hacky functions to tell if on cims, fbook local or fbook cluster
function isUser(user) return sys.execute('echo $USER') == user end
function isFbcode() return package.path:find('.llar') ~= nil end
if isFbcode() then
   require 'fb.trepl'
   package.path = package.path .. ';./?.lua'
end
require 'torch'
require 'nn'
require 'nngraph'
require 'cunn'
pcall(require, 'cudnn')
require 'optim'
require 'pl'
require 'paths'
require 'image'
require 'layers.SpatialConvolutionUpsample'
util = require 'utils.base'
if not isUser('denton') then
  require 'fbnn'
  require 'fbcunn'
  bistro = require 'bistro'
end
--local debugger = require 'fb.debugger'

----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  --lr                (default 0.002)     learning rate
  --beta1             (default 0.5)        momentum term for adam
  --learningRateDecay (default 0.001)
  --weightDecay       (default 0)        weight decay
  --momentum          (default 0.5)        momentum for sgd
  -b,--batchSize      (default 100)        batch size
  -g,--gpu            (default 0)          gpu to use
  --optimizer         (default "adagrad")  "adam" | "sgd" | "adagrad"
  --nEpochs           (default 100)        max training epochs
  --seed              (default 1)          random seed
  --imageSize         (default 96)         image resolution (64 or 96)
  --test                                   test model
  --network           (default "")         network pathname
  --checkpoint        (default "")         restart training from checkpoint
  --reset                                  reset network weights
  --fold              (default 1)          predefined fold for STL-10
  --lrC               (default 1)
  --dataset           (default "stl")            "stl" | "cifar"
]]

if opt.gpu < 0 or opt.gpu > 3 then opt.gpu = false end

if opt.optimizer == "adam" then
  opt.optimizer = optim.adam
elseif opt.optimizer == "sgd" then
  opt.optimizer = optim.sgd
elseif opt.optimizer == "adagrad" then
  opt.optimizer = optim.adagrad
else
  error('Unknown optimizer: ' .. opt.optimizer)
end

print(opt)

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

optimStateD = {
   learningRate = opt.lr,
   learningRateDecay = opt.learningRateDecay,
   beta1 = opt.beta1,
   weightDecay = opt.weightDecay,
   momentum = opt.momentum,
   evalCounter = 0,
}

optimStateC = {
   learningRate = opt.lr,
   learningRateDecay = opt.learningRateDecay,
   beta1 = opt.beta1,
   weightDecay = opt.weightDecay,
   momentum = opt.momentum,
   evalCounter = 0,
}

if opt.test then
  print('loading model ' .. opt.network)
  tmp = torch.load(opt.network)
  netD = tmp.netD
  netC = tmp.netC
elseif opt.checkpoint ~= "" then
  print('loading checkpoint: ' .. opt.checkpoint)
  local tmp = torch.load(opt.checkpoint)
  net = tmp.net
  epoch = tmp.epoch
  optimStateC = tmp.optimStateC
  optimStateD = tmp.optimStateD
else
  -- load model
  local tmp = torch.load(opt.network .. '/model.t7')
  netD = tmp.netD
  netD:remove() --view
  if torch.type(netD.modules[1]):find('ParallelTable') then
    netD:remove(1) -- parallel paths
    netD:remove(1) -- cadd table
  end

  local dummy_in = torch.zeros(opt.batchSize, unpack(opt.geometry))
  local dummy_out = netD:forward(dummy_in:cuda())
  latentDim = dummy_out:nElement()/opt.batchSize
  netC = nn.Sequential()
  netC:add(nn.SpatialDropout())
  netC:add(nn.Reshape(latentDim))
  netC:add(nn.Linear(latentDim, 10))
  netC:add(nn.LogSoftMax())
end

print('netD:')
print(netD)
print('netC:')
print(netC)

-- train from scratch
if not opt.test and opt.reset then
  print('Resetting model weights')
  initModel(netD)
  initModel(netC)
end

local criterion = nn.ClassNLLCriterion()
local confusion = optim.ConfusionMatrix(10)

local x = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
local gen = torch.CudaTensor(opt.batchSize, unpack(opt.geometry))
local targets = torch.CudaTensor(opt.batchSize)

netD:cuda()
netC:cuda()
criterion:cuda()

params_D, grads_D = netD:getParameters()
params_C, grads_C = netC:getParameters()


if not opt.test then
  trainLogger = optim.Logger(opt.network .. '/classification/train.log')
  testLogger = optim.Logger(opt.network .. '/classification/test.log')
end

opt.translate = false
opt.scale = false
opt.small = true -- for stl10
require 'data.data'

function train(dataset)
  netC:training()
  netD:training()
  grads_D:zero()
  grads_C:zero()

  dataset:getBatch(x, targets)
  local out_d = netD:forward(x)
  local out_c = netC:forward(out_d)
  local err = criterion:forward(out_c, targets)
  local dout_c = criterion:backward(out_c, targets)
  local dout_d = netC:backward(out_d, dout_c)
  dout_d:mul(opt.lrC)
  netD:backward(x, dout_d)

  opt.optimizer(function() return 0, grads_D end, params_D, optimStateD)
  opt.optimizer(function() return 0, grads_C end, params_C, optimStateC)

  local top1, top5, nex = classResults(out_c, targets)
  updateConfusion(confusion, out_c, targets)
  return top1
end

function test(dataset, idx)
  dataset:getBatch(x, targets, idx)
  local out_d = netD:forward(x)
  local out_c = netC:forward(out_d)

  local top1, top5, nex = classResults(out_c, targets)
  updateConfusion(confusion, out_c, targets)
  return top1
end

if opt.test then
  opt.translate = false
  opt.scale = false
  print('Evaluating on test set...')
  netD:evaluate()
  netC:evaluate()
  --netC.modules[1]:evaluate()
  assert(torch.type(netC.modules[1]):find('Dropout') ~= nil)
  local top1 = 0
  for i = 1,testData:size(),opt.batchSize do
    xlua.progress(i, testData:size())
    local t1 = test(testData, i)
    top1=top1+t1
  end
  print('\n\ttop1 = ' .. 100*top1/testData:size() .. '%')
  print(confusion)
  confusion:zero()
  os.exit()
end

local best = 0
while true do
  epoch = epoch or 1
  print('\n<trainer> Epoch ' .. epoch .. ' [ lr = ' .. optimStateD.learningRate / (1+optimStateD.evalCounter*optimStateD.learningRateDecay) .. ', momentum = ' .. opt.momentum .. ' ]')
  local top1 = 0
  local nTrain = trainData:size()
  for i = 1,nTrain, opt.batchSize do
    xlua.progress(i, nTrain)
    local t1 = train(trainData)
    top1=top1+t1
  end
  --print(confusionC)
  --confusionC:zero()
  print('\n\ttop1 = ' .. 100*top1/nTrain .. '%')
  trainLogger:add{100*top1/nTrain}
  if 100*top1/nTrain >= 100 then
    quit = true
  end

  print('<tester> Epoch ' .. epoch)
  --net:evaluate()
  local top1 = 0
  local nVal = math.min(4000, valData:size())
  for i = 1,nVal,opt.batchSize do
    xlua.progress(i, nVal)
    local t1 = test(valData, i)
    top1=top1+t1
  end
  print('\n\ttop1 = ' .. 100*top1/nVal .. '%')
  testLogger:add{100*top1/nVal}
  --print(confusionC)
  --
  if optimStateD.momentum > 0 then optimStateD.momentum = optimStateD.momentum + 0.02 end
  if optimStateC.momentum > 0 then optimStateC.momentum = optimStateC.momentum + 0.02 end

  if 100*top1/nVal > best then
    best = 100*top1/nVal
    print('New best: ' .. best)
    print('Saving model: ' .. opt.network .. '/classifier.t7')
    if opt.reset then
      torch.save(opt.network.. '/classifier-' .. opt.fold .. '.t7', {netD = util.sanitize(netD), netC = util.sanitize(netC), optimStateD = optimStateD, optimStateC = optimStateC, epoch = epoch+1})
    else
      torch.save(opt.network.. '/classifier.t7', {netD = util.sanitize(netD), netC = util.sanitize(netC), optimStateD = optimStateD, optimStateC = optimStateC, epoch = epoch+1})
    end
  end

  --confusion:zero()
  util.plotAcc(trainLogger.symbols[1], testLogger.symbols[1], opt.network .. '/classification/acc')

  if quit or epoch > opt.nEpochs then
    print('Best performance on validation set:')
    print('Top1 = ' .. best .. '%')
    break
  end
  epoch = epoch + 1
end

-- test
tmp = torch.load(opt.network .. '/classifier.t7')
netD = tmp.netD
netC = tmp.netC
netD:cuda()
netC:cuda()

opt.translate = false
opt.scale = false
print('Evaluating on test set...')
netD:evaluate()
netC:evaluate()
confusion:zero()
local top1 = 0
for i = 1,testData:size(),opt.batchSize do
  xlua.progress(i, testData:size())
  local t1 = test(testData, i)
  top1=top1+t1
end
print('\n\ttop1 = ' .. 100*top1/testData:size() .. '%')
print(confusion)
confusion:zero()

