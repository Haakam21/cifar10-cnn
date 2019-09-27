import argparse

from google_images_download import google_images_download

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

parser = argparse.ArgumentParser()
parser.add_argument('--classes', nargs='+', choices=class_names + ['all'], default=['all'])
parser.add_argument('--images', type=int, default=5)
parser.add_argument('--format', type=str, default=None)
args = parser.parse_args()

response = google_images_download.googleimagesdownload()

if 'all' in args.classes:
    for _class in class_names:
    	arguments = {'output_directory': 'test-images', 'keywords': _class, 'format': args.format, 'limit': args.images, 'print_urls': True}
else:
    for _class in args.classes:
    	arguments = {'output_directory': 'test-images', 'keywords': _class, 'format': args.format, 'limit': args.images, 'print_urls': True}

try:
    response.download(arguments)
except FileNotFoundError:
    pass
