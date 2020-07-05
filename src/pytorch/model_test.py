import torch
from random import randrange
import conv_block_0

input_size = 100
input_count = 100
output_size = 2

epochs = 100
lr = 0.000001

input = torch.rand((input_count, input_size * input_size)).to('cuda')
input_img = torch.rand((input_count, 3, input_size, input_size)).to('cuda')
# print("input_img: ", input_img)
ideal = torch.zeros(input_count, dtype=torch.long).to('cuda')
for ideal_item in range(input_count):
  ideal[ideal_item] = randrange(output_size)
module = torch.nn.Linear(input_size * input_size, output_size).to('cuda')
loss_fn = torch.nn.CrossEntropyLoss().to('cuda')
cb0 = conv_block_0.make(output_size, (input_size, input_size))

for epoch in range(epochs):
  # pred = module(input)
  pred = conv_block_0.predict(cb0, input_img)
  # print("pred: ", pred)
  # print("ideal: ", ideal)
  loss = loss_fn(pred, ideal)
  print("loss: ", loss)
  loss.backward()
  with torch.no_grad():
    # grad = module.weight.grad.data
    # update = (grad * lr)
    # module.weight.data.sub_(update)
    conv_block_0.back_propagate(cb0, lr=lr, momentum=0.9, wd_part=1)