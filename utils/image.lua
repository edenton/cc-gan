require 'image' 

function random_crop(x, crop)
  assert(x:dim() == 3)
  local crop = math.min(crop, math.min(x:size(2), x:size(3)))
  local sx = math.random(0, x:size(2) - crop)
  local sy = math.random(0, x:size(3) - crop)
  return image.crop(x, sy, sx, sy+crop, sx+crop)
end

function center_crop(x, crop)
  local crop = math.min(crop, math.min(x:size(2), x:size(3)))
  local sx = math.floor((x:size(2) - crop)/2)
  local sy = math.floor((x:size(3) - crop)/2)
  return image.crop(x, sy, sx, sy+crop, sx+crop)
end

function transform(x, size)
  local sz = math.random(x:size(2) - 20, x:size(2))
  local sx = math.random(0, x:size(2) - sz)
  local sy = math.random(0, x:size(3) - sz)
  return image.scale(image.crop(x, sy, sx, sy+sz, sx+sz), size, size)
end

function adjust_meanstd(x, mean, std)
  for c = 1,3 do
    x[c]:add(-mean[c]):div(std[c])
  end
  return x
end

function adjust_contrast(x)
  local img = x
  -- raise S and V to power of 0.25 - 4
  local img_hsv = image.rgb2hsv(img)
  img_hsv[2]:pow(torch.linspace(0.5, 2, 30)[math.random(30)])
  img_hsv[3]:pow(torch.linspace(0.5, 2, 30)[math.random(30)])
  img = image.hsv2rgb(img_hsv)
  -- multiply S and V by factor 0.7-1.4
  local img_hsv = image.rgb2hsv(img)
  img_hsv[2]:mul(torch.linspace(0.7, 1.3, 30)[math.random(30)])
  img_hsv[3]:mul(torch.linspace(0.7, 1.3, 30)[math.random(30)])
  img = image.hsv2rgb(img_hsv)
  -- add to S and V values between −0.1-0.1
  local img_hsv = image.rgb2hsv(img)
  img_hsv[2]:add(torch.linspace(-0.1, 0.1, 30)[math.random(30)])
  img_hsv[3]:add(torch.linspace(-0.1, 0.1, 30)[math.random(30)])
  img = image.hsv2rgb(img_hsv)
  return img
end

function normalize(x, min, max)
  local new_min, new_max = -1, 1
  local old_min, old_max = x:min(), x:max()
  local eps = 1e-7
  x:add(-old_min)
  x:mul(new_max - new_min)
  x:div(old_max - old_min + eps)
  x:add(new_min)
  return x
end
