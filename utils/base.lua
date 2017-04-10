local util = {}

function isNan(x)
  if x:ne(x):sum() > 0 then
    return true
  else
    return false
  end
end

local function merge_table(...)
   local args = {...}
   local t = {}
   for i = #args, 1, -1 do
      for k,v in pairs(args[i]) do
         t[k] = v
      end
   end
   return t
end

function util.plot(x, mask, reverse_mask, noise, fname, N, epoch, opt)
   local N = N or opt.batchSize
   local perimg = 5
   if opt.stochastic then --plot multiple sample of same context if stochastic
      local k = 1
      for i=1,N do
         x[i]:copy(x[k])
         mask[i]:copy(mask[k])
         reverse_mask[i]:copy(reverse_mask[k])
         if i % perimg == 0 then
            k=k+perimg
         end
      end
   end
   local gen = netG:forward({ {x, mask}, noise })
   local to_plot = {}
   for i = 1,N do
      if opt.stochastic then
         if (i-1) % perimg == 0 then
            to_plot[#to_plot+1] = x[i]:float()
            to_plot[#to_plot+1] = torch.cmul(x[i], mask[i]):float()
         end
         to_plot[#to_plot+1] = torch.add(torch.cmul(x[i], mask[i]), torch.cmul(gen[i], reverse_mask[i])):float()
      else
         --to_plot[#to_plot+1] = x[i]:float()
         to_plot[#to_plot+1] = torch.cmul(x[i], mask[i]):float()
         to_plot[#to_plot+1] = gen[i]:float()
         to_plot[#to_plot+1] = torch.add(torch.cmul(x[i], mask[i]), torch.cmul(gen[i], reverse_mask[i])):float()
      end
   end
   local nrow
   if opt.stochastic then
      nrow = 14
   else
      nrow = 15
   end
   local filename = opt.save .. '/gen/' .. epoch .. '_' .. fname .. '.png'
   image.save(filename, image.toDisplayTensor{input=to_plot, scaleeach=true, nrow=nrow})
end

function util.plot_features(x, mask, reverse_mask, fname, N, epoch, opt)
   local N = N or 32 
   local features = netD1:forward(x)
   local gen = netG:forward({features, mask})
   local to_plot = {}
   for i = 1,N do
     for f = 1,opt.nMaps do
       to_plot[#to_plot+1] = features[i][f]:float()
     end
     for f = 1,opt.nMaps do
       to_plot[#to_plot+1] = torch.cmul(features[i][f], mask[i][f]):float()
     end
     for f = 1,opt.nMaps do
       to_plot[#to_plot+1] = gen[i][f]:float()
     end
     for f = 1,opt.nMaps do
       to_plot[#to_plot+1] = torch.add(torch.cmul(features[i][f], mask[i][f]), torch.cmul(gen[i][f], reverse_mask[i][f])):float()
     end
   end
   local filename = opt.save .. '/gen/' .. epoch .. '_' .. fname .. '.png'
   image.save(filename, image.toDisplayTensor{input=to_plot, scaleeach=true, nrow=opt.nMaps})
end

function util.bistro_log(...)
   local tbl = merge_table(...)
   for k, v in pairs(tbl) do
      if type(v) ~= 'number' and type(v) ~= 'string' and type(v) ~= 'boolean' then
         tbl[k] = nil
      end
   end
   local success, bistro = pcall(require, 'bistro')
   if success then
      bistro.EXIT_ON_NAN = false
      bistro.log(tbl)
   end
end

function util.plotTSNE(full_h, labels, fname)
  local tsne = paths.dofile('tsne.lua')

  assert(not isNan(full_h))
  local ydata
  if full_h:size(2) > 2 then
    ydata = tsne(full_h:float())
  else
    ydata = full_h:float()
  end
  if isNan(ydata) then
    print('NaN in tsne results, can\'t plot...')
    return
  end

  local colors= {'#0000FF', '#696969', '#A52A2A', '#FF4500', '#006400', '#8B008B', '#FFD700', '#FF0000', '#000000', '#00FFFF', '#000000'}
  local xs, ys  = {}, {}
  assert(labels:max() <= 11) -- only so many colors XXX: fix this..
  for i = 1,labels:max() do xs[i] = {}; ys[i] = {} end
  local classes = {}
  for n = 1,labels:nElement() do
    local class = labels[n]
    classes[#classes+1] = class
    if class == -1 then class = 11 end
    xs[class][#xs[class] + 1] = ydata[n][1]
    ys[class][#ys[class] + 1] = ydata[n][2]
  end
  Plot = Plot or require 'itorch.Plot'
  local plot
  local started = false
  for c, _ in pairs(xs) do
    if #xs[c] > 0 then
      if not started then
        started = true
        plot = Plot():circle(xs[c], ys[c], colors[c], tostring(c-1))
      else
        plot = plot:circle(xs[c], ys[c], colors[c], tostring(c-1))
      end
    end
  end
  plot:title('tsne on h (epoch ' .. epoch - 1 .. ')')
  plot:legend(true)
  plot:save(fname .. '.html')
end

function util.plotAcc(train, test, fname, title)
  local title = title or ''
  local x = torch.linspace(1, #train, #train)
  local train = torch.Tensor(train)
  local test = torch.Tensor(test)
  Plot = Plot or require 'itorch.Plot'
  local plot = Plot():line(x, train,'red','train accuracy')
  plot = plot:line(x, test, 'blue','test accuracy')
  plot:legend(true)
  plot:title(title)
  plot:save(fname .. '.html')
end

function util.sanitize(net)
   local list = net:listModules()
   for _,val in ipairs(list) do
      for name,field in pairs(val) do
         if torch.type(field) == 'cdata' then val[name] = nil end
         if name == 'homeGradBuffers' then val[name] = nil end
         if name == 'input_gpu' then val['input_gpu'] = {} end
         if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
         if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
         if (name == 'output' or name == 'gradInput') then
            if torch.isTensor(val[name]) then
               val[name] = field.new()
            end
         end
         if  name == 'buffer' or name == 'buffer2' or name == 'normalized'
         or name == 'centered' or name == 'addBuffer' then
            val[name] = nil
         end
      end
   end
   return net
end

function updateConfusion(confusion, output, targets)
  local correct = 0
  for i = 1,targets:nElement() do
    if targets[i] ~= -1 then
      local _, ind = output[i]:max(1)
      confusion:add(ind[1], targets[i])
      if ind[1] == targets[i] then
        correct = correct+1
      end
    end
  end
  return correct
end

function zeroBias(m)
  local name = torch.type(m)
  if name:find('Convolution') then
    m.bias:zero()
  end
end

function classResults(outputs, targets)
  local top1, top5, N = 0, 0, 0
  local _, sorted = outputs:float():sort(2, true)
  for i = 1,opt.batchSize do
    if targets[i] > 0 then -- has label
      N = N+1
      if sorted[i][1] == targets[i] then
        top1 = top1 + 1
      end
      for k = 1,5 do
        if sorted[i][k] == targets[i] then
          top5 = top5 + 1
          break
        end
      end
    end
  end
  return top1, top5, N
end

local function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') or name:find('Linear') then
    m.weight:normal(0.0, 0.02)
    m.bias:fill(0)
  elseif name:find('BatchNormalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias then m.bias:fill(0) end
  end
end

function initModel(model)
  for _, m in pairs(model:listModules()) do
    weights_init(m)
  end
end

function in_array(x, l)
  for k, v in pairs(l) do
    if v == x then return true end
  end
  return false
end

function write_opt(opt)
  local opt_file = io.open(('%s/opt.log'):format(opt.save), 'w')
  for k, v in pairs(opt) do
    opt_file:write(('%s = %s\n'):format(k, v))
  end
  opt_file:close()
end

return util
