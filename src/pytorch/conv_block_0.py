import torch
import math

conv_channels = 64

def get_default_device():
  return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def is_learning_module(module: torch.nn.Module): return hasattr(module, 'weight')

class Skippable(torch.nn.Module):
  def __init__(self, module_list):
    super().__init__()
    self.module = torch.nn.Sequential(*module_list)
  def forward(self, x):
    return x + self.module(x)

class Tap(torch.nn.Module):
  def __init__(self, fn):
    super().__init__()
    self.fn = fn
  def forward(self, x):
    self.fn(x)
    return x

def make(classes_len, img_size, input_channels = 3):
  modules = [
    # Tap(lambda x: print('0 halfs: ', x.size())),
  ]
  conv_channels_i = input_channels
  def add_conv_morf(multiplier, count=1, exact_output_len=False):
    nonlocal modules, conv_channels_i
    morf_len = conv_channels_i * multiplier if not exact_output_len else multiplier
    for i in range(count):
      modules.extend([
        torch.nn.Conv2d(conv_channels_i, morf_len, 3, padding=1, stride=2),
        torch.nn.BatchNorm2d(morf_len),
        torch.nn.ReLU(),
        Skippable([
          torch.nn.Conv2d(morf_len, morf_len, 3, padding=1),
          torch.nn.BatchNorm2d(morf_len),
          torch.nn.ReLU(),
        ]),
      ])
      conv_channels_i = morf_len
  add_conv_morf(conv_channels, 1, True)
  add_conv_morf(2, 7)

  modules.extend([
    # Tap(lambda x: print('8 halfs: ', x.size())),
    torch.nn.Flatten(),
    torch.nn.Linear(conv_channels_i, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, classes_len),
    torch.nn.BatchNorm1d(classes_len),
  ])
  module = torch.nn.Sequential(*modules)
  init(module)
  return module

def init(module, device = None):
  device = device if device is not None else get_default_device()
  module.to(device)

def predict(module, data):
  return module(data)

def calc_weights(module):
  wd_sqr = 0
  for param in module.parameters():
    wd_sqr += param.data.pow(2).sum()
  return wd_sqr

def back_propagate(model, lr, momentum, wd_part = 1):
  for param in model.parameters():
    grad = param.grad.data
    # momentum_val = (model.momentum_val * momentum) + (grad * (1 - momentum)) + 0.0001
    # momentum_sqr_val = (model.momentum_sqr_val * momentum) + ((grad.pow(2)) * (1 - momentum)) + 0.0001
    grad_dims = len(grad.size())
    grad_pick = tuple(torch.zeros(grad_dims, dtype=int).tolist())
    if math.isnan(grad[grad_pick]):
      print("model", model)
      print("grad", grad)
      # print("momentum_val", momentum_val)
    update = (grad * lr)# / momentum_sqr_val.sqrt()) * wd_part

    # model.momentum_val = momentum_val
    # model.momentum_sqr_val = momentum_sqr_val
    param.data.sub_(update)
    model.zero_grad()
