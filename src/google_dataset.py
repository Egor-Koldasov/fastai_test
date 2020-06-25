from simple_image_download import simple_image_download
from settings import google_images_folder

response = simple_image_download.simple_image_download()

donwload_options = {
  "keywords":"Nintendo switch pro controller",
  "limit":20,
  "output_directory": google_images_folder,
  "chromedriver": "/usr/bin/chromedriver",
  "print_urls": True,
}
paths = response.download("Nintendo switch pro controller", 200)
print(paths)
