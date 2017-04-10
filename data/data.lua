assert(not opt.patchMax or opt.patchMax <= opt.imageSize, 'Patch size must be less than image size ')

if opt.dataset == 'stl' then
  require 'data.stl10'
  stl.setScale(opt.imageSize)
  if opt.small then
    labelled_only = true
  else
    labelled_only = false
  end

  trainData = stl.loadTrainSet(labelled_only, opt.fold)
  valData = stl.loadValSet(opt.fold)
  if true then --opt.test then
    testData = stl.loadTestSet()
  end
elseif opt.dataset == 'stl_ms' then
  require 'data.stl10_ms'
  stl.setScale(opt.imageSize)
  if opt.small then
    labelled_only = true
  else
    labelled_only = false
  end

  trainData = stl.loadTrainSet(labelled_only, opt.fold)
  valData = stl.loadValSet(opt.fold)
  if true then --opt.test then
    testData = stl.loadTestSet()
  end
elseif opt.dataset == 'cifar' then
  assert(opt.imageSize == 32, 'Image size must for 32 for STL-10 dataset')
  require 'data.cifar'

  cifar.setScale(opt.imageSize)

  trainData = cifar.loadTrainSet(1,49000, opt.nlabelled)
  valData = cifar.loadTrainSet(49001,50000)
else
  error('Unknown dataset: ' .. opt.dataset)
end
