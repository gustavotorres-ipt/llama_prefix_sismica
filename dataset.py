import json
import os
import random
import torch
from PIL import Image
from config import IMAGE_FOLDER_TRAIN, TEXT_FOLDER_TRAIN, IMAGE_FOLDER_VAL, TEXT_FOLDER_VAL
from torchvision import transforms

class ImageNorm(object):
    def __call__(self, x):
        return (x - x.mean()) / (x.std() + 1e-6)

def read_labels_json(file_path):
    with open(file_path) as f:
        return json.load(f)["label"]


def read_captions_json(file_path):
    with open(file_path) as f:
        captions = json.load(f)["captions"]
        return random.choice(captions)


def load_images(batch_size):
    image_paths = [os.path.join(IMAGE_FOLDER_VAL, filename)
                   for filename in sorted(os.listdir(IMAGE_FOLDER_VAL)) ]
    random.shuffle(image_paths)
    images = [Image.open(path).convert("RGB")
              for path in image_paths[:batch_size]]
    return images


def load_captions(shuffle=False):
    text_paths_train = [os.path.join(TEXT_FOLDER_TRAIN, filename)
                        for filename in sorted(os.listdir(TEXT_FOLDER_TRAIN)) ]
    # captions_train = [read_captions_json(path) for path in text_paths_train]

    text_paths_val = [os.path.join(TEXT_FOLDER_VAL, filename)
                      for filename in sorted(os.listdir(TEXT_FOLDER_VAL)) ]
    # captions_val = [read_captions_json(path) for path in text_paths_val]

    if shuffle:
        random.shuffle(text_paths_train)
        random.shuffle(text_paths_val)

    return text_paths_train, text_paths_val


def load_datasets():
    image_paths_train = [
        os.path.join(IMAGE_FOLDER_TRAIN, filename)
        for filename in sorted(os.listdir(IMAGE_FOLDER_TRAIN))
    ]

    image_paths_val = [
        os.path.join(IMAGE_FOLDER_VAL, filename)
        for filename in sorted(os.listdir(IMAGE_FOLDER_VAL))
    ]

    print("Loading images and captions...")

    captions_train, captions_val = load_captions()

    transformation = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        ImageNorm()
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                     [0.229, 0.224, 0.225])  # ImageNet stats
    ])

    train_dataset = CustomDataset(
        image_paths=image_paths_train, text_paths=captions_train, transform=transformation
    )
    val_dataset = CustomDataset(
        image_paths=image_paths_val, text_paths=captions_val, transform=transformation
    )

    return train_dataset, val_dataset


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, text_paths, transform=None):
        self.image_paths = image_paths
        self.texts = [read_captions_json(path) for path in text_paths]
        self.labels = [read_labels_json(path) for path in text_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text = self.texts[idx]
        label = self.labels[idx]
        return image, text, label
