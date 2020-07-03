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
transforms = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(img_size, pad_if_needed=True), torchvision.transforms.ToTensor()])
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
def make_data_loader(dataset): return torch.utils.data.DataLoader(dataset, batch_size=1)
training_data_loader = make_data_loader(training_subset)
data_i = iter(training_data_loader)
images, classes = next(data_i)


# Model
conv1 = torch.nn.Conv2d(img_channels, 6, 3, padding=1).to(device)
conv2 = torch.nn.Conv2d(6, 12, 3, padding=1).to(device)
conv3 = torch.nn.Conv2d(12, 24, 3, padding=1).to(device)
conv4 = torch.nn.Conv2d(24, 48, 3, padding=1).to(device)
linear1 = torch.nn.Linear(48 * img_size[0] * img_size[1], 128).to(device)
linear2 = torch.nn.Linear(128, len(training_subset.dataset.classes)).to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)

epochs = 10
ln = 0.01

def predict(data):
  conv1_out = torch.nn.functional.relu(conv1(data))
  conv2_out = torch.nn.functional.relu(conv2(conv1_out))
  conv3_out = torch.nn.functional.relu(conv3(conv2_out))
  conv4_out = torch.nn.functional.relu(conv4(conv3_out))
  relu_out_merge = conv4_out.view(-1, linear1.weight.size()[1])
  max_pool_out = torch.nn.functional.max_pool2d(conv4_out, 2)
  linear1_out = linear1(relu_out_merge)
  relu2_out = torch.nn.functional.relu(linear1_out)
  linear2_out = linear2(relu2_out)
  return linear2_out

# learning
def train(data_loader: torch.utils.data.DataLoader, epochs=1, back_propagate=True):
  for epoch_number in range(epochs):
    batch_loss = 0
    for batch_i, data in enumerate(data_loader, 0):

      prediction = predict(data[0].to(device))
      ideal = data[1].to(device)
      loss = criterion(prediction, ideal)

      batch_loss += loss

      print('prediction: ', prediction, '. ideal: ', ideal)
      if back_propagate:
        loss.backward()

        conv1.weight.data.sub_(conv1.weight.grad.data * ln)
        conv2.weight.data.sub_(conv2.weight.grad.data * ln)
        conv3.weight.data.sub_(conv3.weight.grad.data * ln)
        conv4.weight.data.sub_(conv4.weight.grad.data * ln)
        linear1.weight.data.sub_(linear1.weight.grad.data * ln)
        linear2.weight.data.sub_(linear2.weight.grad.data * ln)
      # else:
        # print('prediction: ', prediction, '. ideal: ', ideal)
    print('batch_loss', batch_loss)

train(training_data_loader, epochs=epochs)

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

validation_data_loader = make_data_loader(validation_subset)
  
train(validation_data_loader, back_propagate=False)
