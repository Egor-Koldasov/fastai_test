import os
from pathlib import Path

fileDir = Path(os.path.dirname(os.path.abspath(__file__)))

base_folder_path = fileDir/'..'
google_images_folder = base_folder_path/'google_images'
