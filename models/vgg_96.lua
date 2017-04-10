local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution
local LeakyReLU = nn.LeakyReLU
if opt.backend == 'cudnn' then
   require 'cudnn'
   SpatialConvolution = cudnn.SpatialConvolution
   SpatialFullConvolution = cudnn.SpatialFullConvolution or SpatialFullConvolution -- until em's cudnn is updated
   cudnn.fastest = true
   cudnn.benchmark = true
end

local ngf = opt.ngf or 64
local nc = 3
local inc = 3

netG = nn.Sequential()
local enc = nn.Sequential()
enc:add(nn.CMulTable())
-- input is (nc) x 96 x 96
enc:add(SpatialConvolution(nc, ngf, 4, 4, 2, 2, 1, 1))
enc:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (nf) x 48 x 48
enc:add(SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1))
enc:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (nf) x 24 x 24
enc:add(SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1))
enc:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (nf) x 12 x 12
enc:add(SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1))
enc:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (nf) x 6 x 6

if opt.stochastic then
  local noisep = nn.Sequential()
  -- input is Z: (opt.noiseDim) x 1 x 1
  noisep:add(SpatialFullConvolution(opt.noiseDim, ngf * 8, 6, 6))
  noisep:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
  -- state size: (ngf * 8) x 6 x 6
  local p = nn.ParallelTable()
  p:add(enc)
  p:add(noisep)
  netG:add(p)
  netG:add(nn.JoinTable(2))
  netG:add(SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1))
else
  netG:add(nn.SelectTable(1))
  netG:add(enc)
  netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
end
netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (ngf*4) x 12 x 12
netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 24 x 24
netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 48 x 48
netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size: (nc) x 96 x 96

netD = nn.Sequential()
local orig_path = nn.Sequential()
local gen_path = nn.Sequential()
orig_path:add(nn.CMulTable())
gen_path:add(nn.CMulTable())
netD:add(nn.ParallelTable():add(orig_path):add(gen_path))
netD:add(nn.CAddTable())
-- conv1: 96 -> 48
netD:add(SpatialConvolution(nc, 64, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(64)):add(LeakyReLU(0.2, true))
netD:add(nn.SpatialMaxPooling(2, 2))
-- conv2: 48 -> 24
netD:add(SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(128)):add(LeakyReLU(0.2, true))
netD:add(nn.SpatialMaxPooling(2, 2))
-- conv3: 24 -> 12
netD:add(SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(256)):add(LeakyReLU(0.2, true))
netD:add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(256)):add(LeakyReLU(0.2, true))
netD:add(nn.SpatialMaxPooling(2, 2))
-- conv4: 12 -> 6
netD:add(SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(512)):add(LeakyReLU(0.2, true))
netD:add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(512)):add(LeakyReLU(0.2, true))
netD:add(nn.SpatialMaxPooling(2, 2))
-- conv5: 6 -> 3
netD:add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(512)):add(LeakyReLU(0.2, true))
netD:add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(512)):add(LeakyReLU(0.2, true))
netD:add(nn.SpatialMaxPooling(2, 2))

local c = nn.ConcatTable()
local path_d = nn.Sequential()
path_d:add(SpatialConvolution(512, 1, 3, 3))
path_d:add(nn.Sigmoid())
path_d:add(nn.View(1):setNumInputDims(3))
c:add(path_d)
c:add(nn.Identity())
netD:add(c)

netC = nn.Sequential()
netC:add(nn.SpatialDropout())
netC:add(SpatialConvolution(512, 10, 3, 3))
netC:add(nn.View(10))
netC:add(nn.LogSoftMax())


initModel(netG)
initModel(netD)
