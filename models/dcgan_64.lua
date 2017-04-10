local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution
local LeakyReLU = nn.LeakyReLU
if opt.backend == 'cudnn' then
   require 'cudnn'
   SpatialConvolution = cudnn.SpatialConvolution
   SpatialFullConvolution = cudnn.SpatialFullConvolution 
   SpatialBatchNormalization = cudnn.SpatialBatchNormalization
end

local ngf = opt.ngf or 64
local ndf = opt.ndf or 64
local nc = 3

netG = nn.Sequential()
local enc = nn.Sequential()
enc:add(nn.CMulTable())
-- input is (nc) x 64 x 64
enc:add(SpatialConvolution(nc, ngf, 4, 4, 2, 2, 1, 1))
enc:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (nf) x 32 x 32
enc:add(SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1))
enc:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (nf) x 16 x 16
enc:add(SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1))
enc:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
-- state size: (nf) x 8 x 8
enc:add(SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1))
enc:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (nf) x 4 x 4

if opt.stochastic then
  local noisep = nn.Sequential()
  -- input is Z: (opt.noiseDim) x 1 x 1
  noisep:add(SpatialFullConvolution(opt.noiseDim, ngf * 8, 4, 4))
  noisep:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))
-- state size: (nf) x 4 x 4
  -- state size: (ngf * 8) x 4 x 4
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
-- state size: (ngf*4) x 8 x 8
netG:add(SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))
-- state size: (ngf*2) x 16 x 16
netG:add(SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1))
netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
-- state size: (ngf) x 32 x 32
netG:add(SpatialFullConvolution(ngf, nc, 4, 4, 2, 2, 1, 1))
netG:add(nn.Tanh())
-- state size: (nc) x 64 x 64

netD = nn.Sequential()
local orig_path = nn.Sequential()
local gen_path = nn.Sequential()
orig_path:add(nn.CMulTable())
gen_path:add(nn.CMulTable())
netD:add(nn.ParallelTable():add(orig_path):add(gen_path))
netD:add(nn.CAddTable())
-- input is (nc) x 64 x 64
netD:add(SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf)):add(LeakyReLU(0.2, true))
-- state size: (nf) x 32 x 32
netD:add(SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 2)):add(LeakyReLU(0.2, true))
-- state size: (nf) x 16 x 16
netD:add(SpatialConvolution(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 4)):add(LeakyReLU(0.2, true))
-- state size: (nf) x 8 x 8
netD:add(SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBatchNormalization(ndf * 8)):add(LeakyReLU(0.2, true))
-- state size: (ndf*8) x 4 x 4

local c = nn.ConcatTable()
local path_d = nn.Sequential()
path_d:add(SpatialConvolution(ndf * 8, 1, 4, 4))
path_d:add(nn.Sigmoid())
path_d:add(nn.View(1):setNumInputDims(3))
c:add(path_d)
c:add(nn.Identity())
netD:add(c)

netC = nn.Sequential()
netC:add(SpatialConvolution(ndf * 8, 10, 4, 4))
netC:add(nn.View(10))
netC:add(nn.LogSoftMax())


initModel(netG)
initModel(netD)
initModel(netC)
