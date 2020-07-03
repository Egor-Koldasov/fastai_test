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
      learner.fit_one_cycle(10, max_lr=1e-3)
      learner.save(cacheName, return_path=True)



  print('data.c', data.c)
  learner = cnn_learner(data, models.resnet50, metrics=[error_rate])
  print(learner.summary())
  print('learner.model', learner.model)

  # learner.unfreeze()
  # learner.lr_find()
  # learner.recorder.plot()
  # plot.show()
  trainModel(learner, 'controllers', reset=False)

  interpritation = ClassificationInterpretation.from_learner(learner)
  # interpritation.plot_top_losses(9, figsize=(15,11))
  # plot.show()

  url = 'https://siamesecatstoday.com/wp-content/uploads/2020/04/PIXNIO-1871053-1200x800-570x380.jpg'
  url2 = 'https://i.pinimg.com/originals/7a/06/cf/7a06cfe581b81d314de23731898fffb3.jpg'
  url4 = 'https://i.ytimg.com/vi/tULN2Pfhhzs/maxresdefault.jpg'
  url5 = 'https://www.windowscentral.com/sites/wpcentral.com/files/styles/mediumplus/public/field/image/2018/05/nintendo-switch-pro-controller-wired-communication-hero%20_1_.jpg?itok=E_lLbkuw'
  url6 = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQSqDnTSP7_hvLyDIuLAOzxE2fEEfuZkt-gbg&usqp=CAU'
  url7 = 'https://images.sidelineswap.com/production/005/565/954/09245dcd0006db44_small.jpeg'
  response = requests.get(url7)
  testImage = open_image(BytesIO(response.content))
  testPrediction = learner.predict(testImage)
  print('testPrediction', testPrediction, data.classes[testPrediction[1]])

  # learner.lr_find()
  # learner.recorder.plot()
  # plot.show()

  # learner.unfreeze()
  # learner.fit_one_cycle(5, max_lr=slice(1e-6,1e-5))

  m = learner.model.eval()
  xb, _ = data.one_item(testImage)
  print(xb.shape)
  xb_im = testImage
  xb = xb.cuda()

  def hooked_backward():
    with hook_output(m[0]) as hook_a: 
      preds = m(xb)
    return hook_a
  
  hook_a = hooked_backward()
  acts  = hook_a.stored[0].cpu()
  print(acts.shape)
  avg_acts = acts.mean(0)
  print(avg_acts.shape)
  def show_heatmap(hm):
    _,ax = plt.subplots()
    xb_im.show(ax)
    ax.imshow(hm, alpha=0.6, extent=(0,352,352,0),
              interpolation='bilinear', cmap='magma');
  
  show_heatmap(avg_acts)
  plot.show()



print('__name__', __name__)
if __name__ == '__main__':
  run()