from fastai.vision import *
from fastai.metrics import error_rate
import os
from pathlib import Path
import matplotlib.pyplot as plot
import matplotlib.image as mpimg
import requests
from io import BytesIO


def run():
  bs = 12
  fileDir = Path(os.path.dirname(os.path.abspath(__file__)))
  petsDatasetImagesPath = fileDir/'..'/'datasets'/'pets'/'images'
  dataset_path = fileDir/'..'/'google_images'/'controllers'

  image_file_paths = get_image_files(dataset_path, recurse=True)

  np.random.seed(2)
  path_label_regx = r'/([^/]+)_\d+.jpg$'
  folder_class_regx = r'/controllers/([^/]+)'

  # data = ImageDataBunch.from_name_re(
  #     petsDatasetImagesPath, image_file_paths, path_label_regx,
  #     ds_tfms=get_transforms(),
  #     size=224,
  #     bs=bs
  #   ).normalize(imagenet_stats)

  data = ImageDataBunch.from_folder(
      dataset_path, train=".", valid_pct=0.6,
      ds_tfms=get_transforms(),
      size=224,
      bs=bs
    )
    #.normalize(imagenet_stats)

  # data.show_batch(rows=3, figsize=(7,6))
  plot.show()

  def getModelCachePath(name: str):
    return dataset_path/'models'/f'{name}.pth'

  def trainModel(learner: Learner, cacheName: str, reset=False):
    cachePath = getModelCachePath(cacheName)
    if cachePath.exists() and not reset:
      learner.load(cacheName)
    else:
      learner.fit_one_cycle(3000, max_lr=1e-3)
      learner.save(cacheName, return_path=True)



  learner = cnn_learner(data, models.resnet50, metrics=[error_rate])
  # learner.unfreeze()
  # learner.lr_find()
  # learner.recorder.plot()
  # plot.show()
  trainModel(learner, 'controllers', reset=True)

  interpritation = ClassificationInterpretation.from_learner(learner)
  # interpritation.plot_top_losses(9, figsize=(15,11))
  # plot.show()

  url = 'https://siamesecatstoday.com/wp-content/uploads/2020/04/PIXNIO-1871053-1200x800-570x380.jpg'
  url2 = 'https://i.pinimg.com/originals/7a/06/cf/7a06cfe581b81d314de23731898fffb3.jpg'
  url4 = 'https://i.ytimg.com/vi/tULN2Pfhhzs/maxresdefault.jpg'
  url5 = 'https://www.windowscentral.com/sites/wpcentral.com/files/styles/mediumplus/public/field/image/2018/05/nintendo-switch-pro-controller-wired-communication-hero%20_1_.jpg?itok=E_lLbkuw'
  url6 = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQSqDnTSP7_hvLyDIuLAOzxE2fEEfuZkt-gbg&usqp=CAU'
  url7 = 'https://images.sidelineswap.com/production/005/565/954/09245dcd0006db44_small.jpeg'
  response = requests.get(url4)
  testImage = open_image(BytesIO(response.content))
  testPrediction = learner.predict(testImage)
  print('testPrediction', testPrediction, data.classes[testPrediction[1]])

  # learner.lr_find()
  # learner.recorder.plot()
  # plot.show()

  learner.unfreeze()
  learner.fit_one_cycle(500, max_lr=slice(1e-6,1e-5))

print('__name__', __name__)
if __name__ == '__main__':
  run()