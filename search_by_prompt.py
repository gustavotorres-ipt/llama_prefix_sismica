import torch
import numpy as np
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from numpy._typing import NDArray
from torchvision.transforms.functional import PILImage
from dataset import ImageNorm
from PIL import Image
from config import BATCH_SIZE, device, CUSTOM_CLIP_FILE
from models.clip_model import CLIP_DistilBert_ResNet


IMG_SIZE = 64
BATCH_SIZE = 128
NUMBER_IMAGES_SHOW = 30
MAX_ZEROES_PART = 0.65


def plot_most_similar(closest_images: list[PILImage]):
    """ Present the top NUMBER_IMAGES_SHOW most similar images
    in the screen using matplotlib.
    """
    n_cols = 5
    n_rows = math.ceil(len(closest_images) / n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 10)) # Creates a 2x2 grid

    for i, ax in enumerate(axes.flatten()):
        # Convert PIL Image to NumPy array for Matplotlib display
        ax.imshow(np.array(closest_images[i]))
        ax.axis('off') # Hide axis ticks for cleaner image display
    plt.show()


def split_volume_in_patches(
        seismic_vol: NDArray, transformation: transforms.Compose
        ) -> dict[str, torch.Tensor| list[PILImage]| list[tuple]]:
    """
    Splits a 3D seismic volume into image patches and applies a transformation.

    The function extracts 2D slices (or patches) from a 3D seismic volume,
    converts them into Pillow images, and applies the specified transformation
    pipeline. The transformed images are stacked into a PyTorch tensor while
    the original Pillow images are also returned for visualization or inspection.

    Args:
        seismic_vol (NDArray): A 3D NumPy array representing the seismic volume
            (inline, xline, depth).
        transformation (transforms.Compose): A torchvision transformation
            pipeline applied to each extracted image before stacking them.

    Returns:
        dict[str, torch.Tensor | list[PILImage] | list[tuple]]:
            - "patch_tensors" (torch.Tensor): A tensor containing the
              transformed image patches, (Num_Images, C, H, W).
            - "patch_images" (list[PILImage]): A list containing the original
              Pillow images extracted from the seismic volume.
            - "coordinates" (list[tuple[int, int, int]]): List of 3D coordinates
              corresponding to each extracted patch, indicating its location in
              the seismic volume as (inline, xline, depth).
    """
    patches_tensors = [] # Images in torch tensor format
    patches_images = [] # Images in Pillow format
    coordinates = []

    # Cut the seismic model in multiple patches
    for il in range(0, seismic_vol.shape[0], 10):
        # cut image in 64x64 patches 
        for xl in range(IMG_SIZE, seismic_vol.shape[1], IMG_SIZE):
            for dep in range(IMG_SIZE, seismic_vol.shape[2], IMG_SIZE):
                range_xl = range(xl-IMG_SIZE, xl)
                range_depth = range(dep-IMG_SIZE, dep)

                patch_il = seismic_vol[il,range_xl][:, range_depth].T

                image_patch = Image.fromarray(patch_il).convert('RGB')
                # Count zeroes in image
                percent_zero = (np.count_nonzero(patch_il == 0).sum()
                    / (patch_il.shape[0] * patch_il.shape[1]))
                # Check if more than 80% of image is empty
                if percent_zero > MAX_ZEROES_PART:
                    continue

                # Convert the patch to an image tensor
                tensor_patch = transformation(image_patch)

                patches_images.append(image_patch)
                patches_tensors.append(tensor_patch)
                coordinates.append((il, xl, dep))

    # Convert patches to tensors
    patches_tensors = torch.stack(patches_tensors, dim=0)
    return {'patches_tensors': patches_tensors,
            'patches_images': patches_images,
            'coordinates': coordinates,}


def get_most_similar_images(
        patches_tensors: torch.Tensor, clip_encoder: CLIP_DistilBert_ResNet,
        text_embeds: torch.Tensor) -> NDArray[np.int32]:
    """
    Rank image patches based on their embedding similarity to a text prompt
    embedding and return the indices of the most similar patches.

    The function encodes each image patch using the provided CLIP-based encoder,
    computes a similarity score between the image embeddings and the given text
    embedding, and ranks the patches according to this similarity.

    Args:
        patches_tensors (torch.Tensor): Tensor containing the image patches to
            evaluate. Shape is (N, C, H, W), where N is the number of patches,
            C is the number of channels, and H and W are the height and width
            of each patch.
        clip_encoder (CLIP_DistilBert_ResNet): Model used to generate image
            embeddings compatible with the text embedding space.
        text_embeds (torch.Tensor): Text embedding representing the prompt.
            Shape is (1, D), where D is the embedding dimension of the CLIP model.

    Returns:
        most_similar_imgs (NDArray[np.int32]): Array containing the indices
        of the image patches ranked by similarity ordered from highest to
        lowest similarity.
    """
    cos_similarities = []

    with torch.no_grad():
        # Calc cosine distance between all images
        for i in range(0, patches_tensors.shape[0], BATCH_SIZE):
            img_batch = patches_tensors[i : i+BATCH_SIZE].to(device)

            image_embeds = clip_encoder.encode_image(img_batch)

            text_embeds = F.normalize(text_embeds, dim=1)
            image_embeds = F.normalize(image_embeds, dim=1)

            similarities_batch = image_embeds @ text_embeds.T   # (128, 1)

            cos_similarities.append(similarities_batch.squeeze(1))

        cos_similarities = torch.concat(cos_similarities)
        # Get indexes from most similar patches
        most_similar_imgs = torch.topk(
            cos_similarities, k=NUMBER_IMAGES_SHOW, largest=True, sorted=True,
        ).indices.cpu().detach().numpy().astype(int)

        return most_similar_imgs


def main():
    path = "C:\\Users\\gustavotorres\\Desktop\\dados\\petrobras\\F3\\F3_amplitude.npy"
    seismic_vol = np.load(path)

    # Transform the image to the CLIP format
    transformation = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        ImageNorm()
    ])

    # Transform the image to the CLIP format
    patches_data = split_volume_in_patches(seismic_vol, transformation)
    patches_images = patches_data['patches_images']
    patches_tensors = patches_data['patches_tensors']

    # Load the CLIP model
    clip_encoder = CLIP_DistilBert_ResNet().to(device)
    clip_encoder.load_state_dict(torch.load(CUSTOM_CLIP_FILE))

    clip_encoder.eval()

    # Enter prompt
    prompt = input("What seismic images are you searching? ")
    text_embeds = clip_encoder.encode_text(prompt)

    most_similar_imgs = get_most_similar_images(
        patches_tensors, clip_encoder, text_embeds
    )
    # Show most similar patches on screen
    plot_most_similar([patches_images[i] for i in most_similar_imgs])

    # TODO: Highlight nos patches dos inlines correspondentes e mostrar inlines


if __name__ == "__main__":
    main()
