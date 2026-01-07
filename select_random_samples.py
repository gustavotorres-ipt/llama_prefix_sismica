import os
import random
import shutil

src_train_captions = os.path.join('data', 'captions', 'training')
src_val_captions = os.path.join('data', 'captions', 'validation')
src_train_images = os.path.join('data', 'images', 'training')
src_val_images = os.path.join('data', 'images', 'validation')

dst_train_captions = os.path.join('toy_data', 'captions', 'training')
dst_val_captions = os.path.join('toy_data', 'captions', 'validation')
dst_train_images = os.path.join('toy_data', 'images', 'training')
dst_val_images = os.path.join('toy_data', 'images', 'validation')

os.makedirs(dst_train_captions, exist_ok=True)
os.makedirs(dst_val_captions, exist_ok=True)
os.makedirs(dst_train_images, exist_ok=True)
os.makedirs(dst_val_images, exist_ok=True)

filenames_train = os.listdir(src_train_captions)
random.shuffle(filenames_train)
filenames_train = filenames_train[:1000]

filenames_val = os.listdir(src_val_captions)
random.shuffle(filenames_val)
filenames_val = filenames_val[:100]

filenames_caption_train = [os.path.join(src_train_captions, f) for f in filenames_train]
filenames_image_train = [os.path.join(src_train_images, f.replace(".json", ".png"))
                         for f in filenames_train]

filenames_caption_val = [os.path.join(src_val_captions, f) for f in filenames_val]
filenames_image_val = [os.path.join(src_val_images, f.replace(".json", ".png"))
                       for f in filenames_val]

for img, caption in zip(filenames_image_train, filenames_caption_train):
    shutil.copy(img, dst_train_images)
    shutil.copy(caption, dst_train_captions)
    print(img, "copied.")

for img, caption in zip(filenames_image_val, filenames_caption_val):
    shutil.copy(img, dst_val_images)
    shutil.copy(caption, dst_val_captions)
    print(img, "copied.")
