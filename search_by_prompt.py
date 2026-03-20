import torch
import numpy as np
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cigvis
import cv2
from torchvision import transforms
from numpy._typing import NDArray
from torchvision.transforms.functional import PILImage
from dataset import ImageNorm
from PIL import Image
from config import BATCH_SIZE, device, CUSTOM_CLIP_FILE
from models.clip_model import CLIP_DistilBert_ResNet
from tqdm import tqdm


IMG_SIZE = 64
BATCH_SIZE = 128
NUMBER_IMAGES_SHOW = 30
MAX_ZEROES_PART = 0.65

max_inlines = 0
current_inline = 0

def mouse_callback(event, x, y, flags, param):
    global current_inline
    global max_inlines

    if event == cv2.EVENT_MOUSEWHEEL:
        # The 'flags' parameter in the callback holds the scroll delta information
        # For regular mice, delta is a multiple of 120
        if flags > 0:
            # Scrolled forward (up)
            if current_inline < max_inlines:
                current_inline += 1
        else:
            # Scrolled backward (down)
            if current_inline >= 0:
                current_inline -= 1


def show_volume_opencv(seismic_vol, current_inline, window_name):
    if current_inline < 0:
        current_inline = 0
    if current_inline >= max_inlines:
        current_inline = max_inlines - 1
    img = np.swapaxes(seismic_vol[current_inline], 0, 1)
    img = np.ascontiguousarray(img)

    cv2.putText(img, f"Inline {current_inline}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow(window_name, img)


def opencv_loop(seismic_vol):
    window_name = "Seismic volume"
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    # Main loop
    while True:
        show_volume_opencv(seismic_vol, current_inline, window_name)
        if cv2.waitKey(1) & 0xFF == 27: # Press 'Esc' to exit
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break


def plot_cigvis(volume, colormap="Greys"):
    # volume em formato numpy 3d
    volume = volume.max() - volume
    nodes = cigvis.create_slices(volume, cmap=colormap)

    # Visualize in 3D
    cigvis.plot3D(nodes)


def plot_most_similar(closest_images: list[PILImage]):
    """ Present the top NUMBER_IMAGES_SHOW most similar images
    in the screen using matplotlib.
    """
    n_cols = 5
    n_rows = math.ceil(len(closest_images) / n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 10))

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

    print("Slicing seismic volume...")
    # Cut the seismic model in multiple patches
    for il in tqdm(range(0, seismic_vol.shape[0])):
        # cut image in 64x64 patches 
        for xl in range(IMG_SIZE, seismic_vol.shape[1], IMG_SIZE):
            for dep in range(IMG_SIZE, seismic_vol.shape[2], IMG_SIZE):
                x1 = xl - IMG_SIZE
                x2 = xl
                y1 = dep - IMG_SIZE
                y2 = dep

                patch_il = seismic_vol[il, x1:x2, y1:y2].T

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
    embedding and return the indices of the most similar patches: the
    75-perncetile most similar.

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

    print("Calculating most similar images...")

    with torch.no_grad():
        # Calc cosine distance between all images
        for i in tqdm(range(0, patches_tensors.shape[0], BATCH_SIZE)):
            img_batch = patches_tensors[i : i+BATCH_SIZE].to(device)

            image_embeds = clip_encoder.encode_image(img_batch)

            text_embeds = F.normalize(text_embeds, dim=1)
            image_embeds = F.normalize(image_embeds, dim=1)

            similarities_batch = image_embeds @ text_embeds.T   # (128, 1)

            cos_similarities.append(similarities_batch.squeeze(1))

        cos_similarities = torch.concat(cos_similarities)

        percentile = torch.quantile(cos_similarities, 0.9)
        selected_imgs = cos_similarities >= percentile

        most_similar_imgs = [i for i in range(len(selected_imgs)) if selected_imgs[i]]

        return most_similar_imgs 


def highlight_volume(seismic_vol, coordinates):
    seismic_vol_copy = np.copy(seismic_vol)

    # Clip the seismic volume under 1 and over 99 quantiles
    quantiles = np.quantile(seismic_vol, [0.01, 0.99])
    seismic_vol_copy[seismic_vol > quantiles[1]] = quantiles[1]
    seismic_vol_copy[seismic_vol < quantiles[0]] = quantiles[0]

    seismic_vol_copy = (255 * 
        (seismic_vol_copy - seismic_vol_copy.min()) /
        (seismic_vol_copy.max() - seismic_vol_copy.min())
    ).astype(np.uint8)

    seismic_vol_copy = np.stack(
        (seismic_vol_copy, seismic_vol_copy, seismic_vol_copy),
        axis=-1)

    for il, xl, d in coordinates:
        x1 = xl - IMG_SIZE
        x2 = xl
        y1 = d - IMG_SIZE
        y2 = d
        seismic_vol_copy[il, x1:x2, y1:y2, 2] = 180  #1.5 * seismic_vol[il, y1:y2, x1:x2]
    return seismic_vol_copy


def main():
    path = "C:\\Users\\gustavotorres\\Desktop\\dados\\petrobras\\F3\\F3_amplitude.npy"
    seismic_vol = np.load(path).astype(np.float32)

    global max_inlines
    max_inlines = seismic_vol.shape[0]

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

    while(True):
        # Enter prompt
        prompt = input("What seismic images are you searching? (type 'end' to stop) ")

        if prompt.lower() == 'end':
            break
        text_embeds = clip_encoder.encode_text(prompt)

        most_similar_imgs = get_most_similar_images(
            patches_tensors, clip_encoder, text_embeds
        )

        coordinates = patches_data['coordinates']
        coordinates = [coordinates[i] for i in most_similar_imgs]

        seismic_vol_3channels = highlight_volume(seismic_vol, coordinates)  #1.5 * seismic_vol[il, y1:y2, x1:x2]

        opencv_loop(seismic_vol_3channels)

        # cv2.waitKey(0)
        cv2.destroyAllWindows() 


if __name__ == "__main__":
    main()
