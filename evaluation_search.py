import numpy as np
import torch
import torch.nn.functional as F
from numpy._typing import NDArray
from tqdm import tqdm
from models.clip_model import CLIP_DistilBert_ResNet
from config import CUSTOM_CLIP_FILE, BATCH_SIZE, device
from torch.utils.data import DataLoader
from dataset import load_datasets


def get_similar_images(
    dataloader: DataLoader, clip_encoder: CLIP_DistilBert_ResNet,
    prompt_embeds: torch.Tensor, num_images = 150
) -> NDArray[np.int32]:

    cos_similarities_torch = []
    labels = []

    print("Calculating most similar images...")

    with torch.no_grad():
        # Calc cosine distance between all images
        for img_batch, _, label_batch in tqdm(dataloader):
            img_batch = img_batch.to(device)
            labels += label_batch

            image_embeds = clip_encoder.encode_image(img_batch)

            prompt_embeds = F.normalize(prompt_embeds, dim=1)
            image_embeds = F.normalize(image_embeds, dim=1)

            similarities_batch = image_embeds @ prompt_embeds.T   # (128, 1)

            cos_similarities_torch.append(similarities_batch.squeeze(1))

        cos_similarities_torch = torch.concat(cos_similarities_torch)
        cos_similarities = cos_similarities_torch.detach().cpu().numpy()

        most_similar_imgs = np.argsort(cos_similarities)[::-1][:num_images]
        labels = np.array(labels)

        #percentile = torch.quantile(cos_similarities, 0.9)
        #selected_imgs = cos_similarities >= percentile

        return labels[most_similar_imgs]


def main():
    _, val_dataset = load_datasets()

    dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load the CLIP model
    clip_encoder = CLIP_DistilBert_ResNet().to(device)
    clip_encoder.load_state_dict(torch.load(CUSTOM_CLIP_FILE))

    clip_encoder.eval()

    target_label = input("What is the label? ")
    text_embeds = clip_encoder.encode_text(f"{target_label} seismic facies")

    labels_top_images = get_similar_images(dataloader, clip_encoder, text_embeds)
    valid_labels = [label for label in labels_top_images if label == target_label]

    accuracy = len(valid_labels) / len(labels_top_images)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    main()
