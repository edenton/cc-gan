require 'image'
require 'utils.image'

stl = {}
-- data paths
stl.path_trainset = opt.dataPath ..  'stl10-train.t7'
stl.path_testset = opt.dataPath .. 'stl10-test.t7'
stl.path_unlabelled = opt.dataPath .. 'stl10-unlabeled.t7'

stl.scale = 64

function stl.transform(x)
  local img, sz
  if opt and opt.scale then
    if stl.scale == 64 then
      sz = math.random(64, 96)
    else
      sz = math.random(90, 96)
    end
  else
    sz = stl.scale
  end
  if opt and opt.translate then --random crop
    img = random_crop(x, sz)
  else -- center crop
    img = center_crop(x, sz)
  end
  img = image.scale(img, stl.scale, stl.scale)
  if opt and opt.color then
    img = adjust_contrast(img)
  end

  --return img
  return normalize(img)
end

function stl.setScale(scale)
  stl.scale = scale
end

function stl.loadTrainSet(labelled, fold)
  local fold = fold or 1
  local ind = torch.load(stl.path_dataset .. 'indices.t7') -- folds for stl10
  if labelled then
    return stl.loadDataset({stl.path_trainset}, ind[fold])
  else
    return stl.loadDataset({stl.path_trainset, stl.path_unlabelled}, ind[fold])
  end
end

function stl.loadValSet(fold)
  local fold = fold or 1
  local ind = torch.load(stl.path_dataset .. 'indices.t7') -- folds for stl10
  -- get all indices not in ind[fold]
  local use = {}
  for i = 1,5000 do
    if ind[fold]:eq(i):sum() == 0 then
      use[#use+1] = i
    end
  end
  return stl.loadDataset({stl.path_trainset}, torch.LongTensor(use))
end

function stl.loadTestSet()
   return stl.loadDataset({stl.path_testset})
end

function stl.loadDataset(fileName, labelled_idx)
  local data
  local labels
  if #fileName == 1 then -- load only one file
    local f = torch.load(fileName[1])
    data = f.data:float()
    labels = f.label
    if labelled_idx then
      data = data:index(1, labelled_idx)
      labels = labels:index(1, labelled_idx)
    end
  else -- load 2 files and concatenate
    local f = torch.load(fileName[1])
    local nl = labelled_idx:size(1)
    local nu = 100000
    data = torch.FloatTensor(nl+nu, 3, 96, 96)
    labels = torch.FloatTensor(nl+nu):fill(-1)
    if labelled_idx then
      data:sub(1,nl):copy(f.data:index(1, labelled_idx))
      labels:sub(1,nl):copy(f.label:index(1, labelled_idx))
    end
    local f = torch.load(fileName[2])
    data:sub(nl+1,nl+nu):copy(f.data)
  end
  collectgarbage()

  local dataset = {}
  dataset.data = data
  dataset.labels = labels
  local N = data:size(1)
  local labelledN = labels:ne(-1):sum()
  print('<stl10> loaded ' .. N .. ' examples ('  .. labelledN .. ' labels)')

  function dataset:size()
    return N
  end

  function dataset:setMeanStd(mean, std)
    if not (mean and std) then
      self.mean, self.std = self:computeMeanStd()
    else
      self.mean = mean
      self.std = std
    end
  end

  function dataset:computeMeanStd(nex)
    local nex = nex or 10000
    local mean = torch.Tensor(3):zero()
    local std = torch.Tensor(3):zero()
    for i = 1, nex do
      local x = stl.transform(self.data[math.random(N)])
      for c = 1,3 do
        mean[c] = mean[c] + x[c]:mean()
      end
    end
    mean:div(nex)
    for i = 1, nex do
      local x = stl.transform(self.data[math.random(N)])
      for c = 1,3 do
        std[c] = std[c] + x[c]:add(-mean[c]):std()
      end
    end
    std:div(nex)
    print('mean = ' .. mean[1] .. ' ' .. mean[2] .. ' ' .. mean[3])
    print('std =  ' .. std[1] .. ' ' .. std[2] .. ' ' .. std[3])
    self.mean = mean
    self.std = std
    return mean, std
  end

  function dataset:normalize(mean, std)
    local mean = mean or {}
    local std = std or {}
    for c = 1,3 do
      if not mean[c] then
        mean[c] = self.data[{ {1,N}, c}]:mean()
        std[c] = self.data[{ {1,N}, c}]:std()
      end
      self.data[{ {1,N}, c}]:add(-mean[c])
      self.data[{ {1,N}, c}]:mul(1/std[c])
    end
    return mean, std
  end

  function dataset:getBatch(x, labels, sidx, nlabelled)
    for n = 1,x:size(1) do
      local idx
      if sidx then -- option to get data in order instead of random
        idx = sidx+n-1
      elseif nlabelled and labelledN < N then
        if n <= nlabelled then
          idx = math.random(labelledN)
        else
          idx = math.random(labelledN+1, N)
        end
      else
        idx = math.random(N)
      end
      local img
      if torch.uniform() > 0.5 then
        img = image.hflip(self.data[idx])
      else
        img = self.data[idx]
      end
      --[[
      x[n]:copy(adjust_meanstd(stl.transform(img), self.mean, self.std))
      --]]
      x[n]:copy(stl.transform(img))
      if labels then
        labels[n] = self.labels[idx]
      end
    end
  end

  function dataset:plotData(fname)
    local to_plot = {}
    for i = 1,10 do
      for j = 1,20 do
        to_plot[#to_plot+1] = stl.transform(self.data[i])
      end
    end
    image.save(fname,image.toDisplayTensor{input=to_plot, scaleeach=true, nrow=20})
  end

  return dataset
end
