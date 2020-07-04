import torch
import torchvision
import time
import PIL
import os
from io import BytesIO
import requests
import matplotlib.pyplot as plt
import numpy
from typing import Union
from pathlib import Path

# class Model0(torch.nn.Module):

mock_img = torch.rand(1, 3, 16, 16)
classes_mock = torch.zeros(32)
classes_mock[10] = 1

#params
img_size = (256, 256)
img_size = (128, 128)
img_channels = 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# dataset
fileDir = Path(os.path.dirname(os.path.abspath(__file__)))
controller_dataset_path = fileDir/".."/".."/"google_images"/"controllers"
transforms = torchvision.transforms.Compose([
  torchvision.transforms.RandomCrop(img_size, pad_if_needed=True),
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
image_folder_dataset = torchvision.datasets.ImageFolder(controller_dataset_path, transform=transforms)
print(len(image_folder_dataset.imgs))
img_len = len(image_folder_dataset.imgs)
training_len =  round(img_len * 0.8)
validation_len =  img_len - training_len
training_subset, validation_subset = \
  torch.utils.data.random_split(
    image_folder_dataset, (training_len, validation_len)
  )
print(training_subset.dataset.classes)
def make_data_loader(dataset): return torch.utils.data.DataLoader(dataset, batch_size=16)
training_data_loader = make_data_loader(training_subset)
data_i = iter(training_data_loader)
images, classes = next(data_i)


# Model
conv1 = torch.nn.Conv2d(img_channels, 6, 3, padding=1).to(device)
conv2 = torch.nn.Conv2d(6, 9, 5, padding=2).to(device)
conv3 = torch.nn.Conv2d(9, 15, 7, padding=3).to(device)
conv4 = torch.nn.Conv2d(15, 25, 9, padding=4).to(device)
linear1 = torch.nn.Linear(25 * img_size[0] * img_size[1], 50).to(device)
linear2 = torch.nn.Linear(50, len(training_subset.dataset.classes)).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)

epochs = 500
lr = 0.03
wd = 0.01

# learning
def predict(data):
  conv1_out = torch.nn.functional.relu(conv1(data))
  # print("conv1_out", conv1_out)
  conv2_out = torch.nn.functional.relu(conv2(conv1_out))
  # print("conv2_out", conv2_out)
  conv3_out = torch.nn.functional.relu(conv3(conv2_out))
  # print("conv3_out", conv3_out)
  conv4_out = torch.nn.functional.relu(conv4(conv3_out))
  # print("conv4_out", conv4_out)
  relu_out_merge = conv4_out.view(-1, linear1.weight.size()[1])
  linear1_out = linear1(relu_out_merge)
  # print("linear1_out", linear1_out)
  relu2_out = torch.nn.functional.relu(linear1_out)
  # print("relu2_out", relu2_out)
  linear2_out = linear2(relu2_out)
  # print("linear2_out", linear2_out)
  return linear2_out

def class_loss(pred: torch.Tensor, ideal: torch.Tensor):
  # (c) tensor
  pred_size = pred.size()
  # number of classes times number of items in the batch
  losses = torch.empty(pred_size[0] * pred_size[1], requires_grad=True).to(device)
  for (one_pred_i, one_pred) in enumerate(pred, 0):
    for (pred_class, pred_class_guess) in enumerate(one_pred, 0):
      ideal_class = ideal[one_pred_i]
      ideal_guess = 1 if ideal_class == pred_class else 0
      loss = (ideal_guess - pred_class_guess) ** 2
      losses[(one_pred_i * len(one_pred)) + pred_class] = loss
  return losses.mean()
    
def sum_of_squares(tensor: torch.Tensor): tensor.pow(2).sum()

weight_models = [
  conv1,
  conv2,
  conv3,
  conv4,
  linear1,
  linear2,
]

def do_one_epoch(data_loader: torch.utils.data.DataLoader, back_propagate=True):
  batch_loss = 0
  item_count = 0
  item_correct = 0
  losses = torch.empty(len(data_loader))
  # for weight_model in weight_models:
  #   if weight_model.momented2 == NoValue: weight_model.momented2 = 1
  #   wd_part = wd * w2
  for batch_i, data in enumerate(data_loader, 0):
    prediction = predict(data[0].to(device))
    prediction_classes = prediction.argmax(dim=1)
    ideal = data[1].to(device)
    loss = class_loss(prediction, ideal)
    losses[batch_i] = loss

    for (item_i, pred_class) in enumerate(prediction_classes, 0):
      if pred_class == ideal[item_i]: item_correct += 1
      item_count += 1

    batch_loss += loss

    # if batch_i % 50 == 0:
    #   print('prediction: ', prediction, '. ideal: ', ideal)
    if back_propagate:
      loss.backward()
      w2 = 0.
      with torch.no_grad():
        for weight_model in weight_models:
          w2 += weight_model.weight.data.pow(2).sum()

        for weight_model in weight_models:
          wd_part = wd * w2
          update = weight_model.weight.grad.data * lr * wd_part
          weight_model.weight.data.sub_(update)
          weight_model.zero_grad()
    # else:
      # print('prediction: ', prediction, '. ideal: ', ideal)
  mean_loss = torch.mean(losses)
  return (mean_loss, item_correct / item_count)

validation_data_loader = make_data_loader(validation_subset)
def train(training_dl: torch.utils.data.DataLoader, validation_dl: torch.utils.data.DataLoader, epochs=1):
  min_loss = 9999
  last_loss = min_loss
  for epoch_number in range(epochs):
    train_res = do_one_epoch(training_dl)
    with torch.no_grad():
      validation_res = do_one_epoch(validation_dl, back_propagate=False)
    number_format = "{:6.4f}"
    is_at_min = train_res[0] < min_loss
    is_decreasing = train_res[0] < last_loss
    if is_at_min: min_loss = train_res[0]
    last_loss = train_res[0]
    loss_sign = '-' if is_at_min else '~' if is_decreasing else ' '
    tl = number_format.format(train_res[0])
    tv = number_format.format(train_res[1])
    vl = number_format.format(validation_res[0])
    vv = number_format.format(validation_res[1])
    print(f"t_loss: {tl} {loss_sign}\t\tt_valid: {tv}\t\tv_loss: {vl}\t\tv_valid: {vv}")

train(training_data_loader, validation_data_loader, epochs=epochs)

# testing
def load_image_PIL(url: str):
  response = requests.get(url)
  byte_buffer = BytesIO(response.content)
  pil_image = PIL.Image.open(byte_buffer)
  return pil_image

def show_img(img: Union[torch.Tensor, PIL.Image.Image]):
  img_tensor = img if isinstance(img, torch.Tensor) else torchvision.transforms.ToTensor()(img)
  plt.imshow(numpy.transpose(img_tensor.numpy(), (1, 2, 0)))
  plt.show()

nintendo_pro_url1 = 'https://www.windowscentral.com/sites/wpcentral.com/files/styles/mediumplus/public/field/image/2018/05/nintendo-switch-pro-controller-wired-communication-hero%20_1_.jpg?itok=E_lLbkuw'
xbox_url = 'https://images.sidelineswap.com/production/005/565/954/09245dcd0006db44_small.jpeg'

test_img_PIL = load_image_PIL(xbox_url)
test_img_tensor = transforms(test_img_PIL)

test_image_prediction = predict(test_img_tensor[None].to(device))
print(test_image_prediction)
max_pred = test_image_prediction.argmax()
predicted_class_label = training_subset.dataset.classes[max_pred]
print("prediction:", predicted_class_label)

  
do_one_epoch(validation_data_loader, back_propagate=False)
