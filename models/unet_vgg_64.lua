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

local x = nn.Identity()()
local z = nn.Identity()()

local xc = nn.CMulTable()(x)
-- nc x 64 x 64
local enc1 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(ngf)(SpatialConvolution(nc, ngf, 4, 4, 2, 2, 1, 1)(xc)))
-- ngf x 32 x 32
local enc2 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(ngf*2)(SpatialConvolution(ngf, ngf*2, 4, 4, 2, 2, 1, 1)(enc1)))
-- ngf*2 x 16 x 16 
local enc3 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(ngf*4)(SpatialConvolution(ngf*2, ngf*4, 4, 4, 2, 2, 1, 1)(enc2)))
-- ngf*4 x 8 x 8 
--
-- noiseDim x 1 x 1
local noise1 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(ngf)(SpatialFullConvolution(opt.noiseDim, ngf, 8, 8)(z)))
-- ngf x 8 x 8

local join = nn.JoinTable(2)({enc3, noise1})

-- ngf*2+ngf x 8 x 8
local enc4 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(ngf*8)(SpatialConvolution(ngf*4+ngf, ngf*8, 4, 4, 2, 2, 1, 1)(join)))
-- ngf*8 x 4 x 4
local dec1 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(ngf*4)(SpatialFullConvolution(ngf*8, ngf*4, 4, 4, 2, 2, 1, 1)(enc4)))
-- ngf*4 x 8 x 8
local dec2 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(ngf*2)(SpatialFullConvolution(ngf*4*2, ngf*2, 4, 4, 2, 2, 1, 1)( nn.JoinTable(2)({dec1, enc3} ))))
-- ngf*2 x 16 x 16
local dec3 = nn.LeakyReLU(0.2, true)(SpatialBatchNormalization(ngf)(SpatialFullConvolution(ngf*2*2, ngf, 4, 4, 2, 2, 1, 1)( nn.JoinTable(2)({dec2, enc2} ))))
-- ngf x 32 x 32
local dec4 = nn.Tanh()(SpatialFullConvolution(ngf*2, nc, 4, 4, 2, 2, 1, 1)( nn.JoinTable(2)({dec3, enc1})))
-- nc x 64 x 64
netG = nn.gModule({x, z}, {dec4})


netD = nn.Sequential()
local orig_path = nn.Sequential()
local gen_path = nn.Sequential()
orig_path:add(nn.CMulTable())
gen_path:add(nn.CMulTable())
netD:add(nn.ParallelTable():add(orig_path):add(gen_path))
netD:add(nn.CAddTable())
-- conv1: 64 -> 32
netD:add(SpatialConvolution(nc, 64, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(64)):add(LeakyReLU(0.2, true))
netD:add(nn.SpatialMaxPooling(2, 2))
-- conv2: 32 -> 16
netD:add(SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(128)):add(LeakyReLU(0.2, true))
netD:add(nn.SpatialMaxPooling(2, 2))
-- conv3: 16 -> 8
netD:add(SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(256)):add(LeakyReLU(0.2, true))
netD:add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(256)):add(LeakyReLU(0.2, true))
netD:add(nn.SpatialMaxPooling(2, 2))
-- conv4: 8 -> 4
netD:add(SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(512)):add(LeakyReLU(0.2, true))
netD:add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(512)):add(LeakyReLU(0.2, true))
netD:add(nn.SpatialMaxPooling(2, 2))
-- conv5: 4 -> 2
netD:add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(512)):add(LeakyReLU(0.2, true))
netD:add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
netD:add(SpatialBatchNormalization(512)):add(LeakyReLU(0.2, true))
netD:add(nn.SpatialMaxPooling(2, 2))

local c = nn.ConcatTable()
local path_d = nn.Sequential()
path_d:add(SpatialConvolution(512, 1, 2, 2))
path_d:add(nn.Sigmoid())
path_d:add(nn.View(1):setNumInputDims(3))
c:add(path_d)
c:add(nn.Identity())
netD:add(c)

netC = nn.Sequential()
netC:add(nn.SpatialDropout())
netC:add(SpatialConvolution(512, 10, 2, 2))
netC:add(nn.View(10))
netC:add(nn.LogSoftMax())


initModel(netG)
initModel(netD)
initModel(netC)
