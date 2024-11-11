import torch
import torchvision
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
import requests
import os
from tqdm import tqdm

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(local_filename, 'wb') as f, tqdm(
            desc=local_filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in r.iter_content(chunk_size=8192):
                size = f.write(chunk)
                progress_bar.update(size)

def load_dataset():
    # Download COCO 2017 val images and annotations
    image_url = "http://images.cocodataset.org/zips/val2017.zip"
    annotation_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

    if not os.path.exists('val2017.zip'):
        print("Downloading COCO val2017 images...")
        download_file(image_url, 'val2017.zip')

    if not os.path.exists('annotations_trainval2017.zip'):
        print("Downloading COCO annotations...")
        download_file(annotation_url, 'annotations_trainval2017.zip')

    # Extract files
    print("Extracting files...")
    !unzip -q -o val2017.zip
    !unzip -q -o annotations_trainval2017.zip

    # Load COCO dataset
    annotation_file = 'annotations/instances_val2017.json'
    image_dir = 'val2017'

    coco = COCO(annotation_file)
    cat_ids = coco.getCatIds(['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)

    # Let's use only the first 100 images for this example
    img_ids = img_ids[:100]

    dataset = CocoDetection(image_dir, annotation_file,
                            transform=torchvision.transforms.ToTensor())

    # Filter dataset to only include images with persons
    filtered_dataset = torch.utils.data.Subset(dataset, img_ids)

    return filtered_dataset, coco

# Usage
dataset, coco = load_dataset()
print(f"Dataset size: {len(dataset)}")
