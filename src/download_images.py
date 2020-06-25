from fastai.vision.data import download_images
from settings import base_folder_path

def download_images_from_file(file_name: str):
  download_images(base_folder_path/'urls'/file_name, base_folder_path/'google_images'/file_name)

download_images_from_file('xbox_one_controller')