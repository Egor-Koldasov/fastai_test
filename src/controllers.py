from fastai.vision import *
from fastai.metrics import error_rate
from fastai.callbacks.hooks import *
import os
from pathlib import Path
import matplotlib.pyplot as plot
import matplotlib.image as mpimg
import requests
from io import BytesIO

def run():
  lr = 0.001
  bs = 12
  epoch_count = 20

  fileDir = Path(os.path.dirname(os.path.abspath(__file__)))
  dataset_path = fileDir/'..'/'google_images'/'controllers'

  np.random.seed(2)

  data = ImageDataBunch.from_folder(
      dataset_path, train=".", valid_pct=0.2,
      ds_tfms=get_transforms(),
      size=224,
      bs=bs
    ).normalize(imagenet_stats)

  plot.show()

  def getModelCachePath(name: str):
    return dataset_path/'models'/f'{name}.pth'

  def trainModel(learner: Learner, cacheName: str, reset=False):
    cachePath = getModelCachePath(cacheName)
    if cachePath.exists() and not reset:
      learner.load(cacheName)
    else:
      learner.fit_one_cycle(epoch_count, max_lr=lr)
      learner.save(cacheName, return_path=True)



  print('data.c', data.c)
  learner = cnn_learner(data, models.resnet50, metrics=[error_rate])

  trainModel(learner, 'controllers', reset=True)


print('__name__', __name__)
if __name__ == '__main__':
  run()