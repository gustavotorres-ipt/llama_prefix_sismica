import json
import os
import random
import torch
from PIL import Image
from config import IMAGE_FOLDER_TRAIN, TEXT_FOLDER_TRAIN, IMAGE_FOLDER_VAL, TEXT_FOLDER_VAL
from torchvision import transforms

def read_captions_json(file_path):
    with open(file_path) as f:
        captions = json.load(f)["captions"]
        return captions[0] #random.choice(captions)


def load_images(batch_size):
    image_paths = [os.path.join(IMAGE_FOLDER_VAL, filename)
                   for filename in os.listdir(IMAGE_FOLDER_VAL) ]
    random.shuffle(image_paths)
    images = [Image.open(path).convert("RGB")
              for path in image_paths[:batch_size]]
    return images


def load_captions(shuffle=False):
    text_paths_train = [os.path.join(TEXT_FOLDER_TRAIN, filename)
                        for filename in os.listdir(TEXT_FOLDER_TRAIN) ]
    captions_train = [read_captions_json(path) for path in text_paths_train]

    text_paths_val = [os.path.join(TEXT_FOLDER_VAL, filename)
                      for filename in os.listdir(TEXT_FOLDER_VAL) ]
    captions_val = [read_captions_json(path) for path in text_paths_val]

    if shuffle:
        random.shuffle(captions_train)
        random.shuffle(captions_val)

    return captions_train, captions_val


def load_datasets():
    image_paths_train = [
        os.path.join(IMAGE_FOLDER_TRAIN, filename)
        for filename in os.listdir(IMAGE_FOLDER_TRAIN)
    ]

    image_paths_val = [
        os.path.join(IMAGE_FOLDER_VAL, filename)
        for filename in os.listdir(IMAGE_FOLDER_VAL)
    ]

    print("Loading images and captions...")

    captions_train, captions_val = load_captions()

    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # ImageNet stats
    ])

    train_dataset = CustomDataset(
        image_paths=image_paths_train, texts=captions_train, transform=transformation
    )
    val_dataset = CustomDataset(
        image_paths=image_paths_val, texts=captions_val, transform=transformation
    )

    return train_dataset, val_dataset


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, texts, transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text = self.texts[idx]
        return image, text
