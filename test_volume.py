import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers.models.auto.modeling_tf_auto import TF_MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES
from models.llama_prefix import LlamaPrefix
from models.clip_model import CLIP_DistilBert_ResNet
from config import CUSTOM_CLIP_FILE, device, LANG_PREFIX_CHECKPOINT
from seismicloader import carregar_arquivo
from torchvision import transforms
from tqdm import tqdm

torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.benchmark = False

# DATA_VOLUME_PATH = 'C:/Users/gustavotorres/Desktop/dados/petrobras/Waka3D/WAKA_3D_null.sgy'
DATA_VOLUME_PATH = 'C:/Users/gustavotorres/Desktop/dados/petrobras/Parihaka-labeled/data_train.npy'
# DATA_VOLUME_PATH = 'C:/Users/gustavotorres/Desktop/dados/petrobras/F3/F3_amplitude.npy'
OUTPUT_FILENAME = 'parihaka_analysis.txt'
INLINE_SKIP = 10

REF_CAPTIONS = {
    'chaotic': 'chaotic pattern',
    'divergent': 'divergent seismic',
    'parallel': 'parallel-bedded',
    'sigmoid': 'oblique clinoform',
}


def calc_cosine_similarities(model, input_text):
    input_ids = model.tokenize_texts(input_text).input_ids
    text_embeds = model.lang_model.model.embed_tokens(input_ids)
    mean_embeds_text = text_embeds.mean(dim=1)

    labels_similarities = {}

    for label, caption in REF_CAPTIONS.items():
        ref_ids = model.tokenize_texts(caption).input_ids
        caption_embeds = model.lang_model.model.embed_tokens(ref_ids)
        mean_embeds_caption = caption_embeds.mean(dim=1)

        cos_sim = F.cosine_similarity(
            mean_embeds_text, mean_embeds_caption).item()

        labels_similarities[label] = cos_sim
    return labels_similarities


class ImageNorm(object):
    def __call__(self, x):
        return (x - x.mean()) / (x.std() + 1e-6)

def cut_text_after_last_period(text: str) -> str:
    idx = text.rfind(".")
    if idx == -1:
        return text
    return text[:idx] + '.'


def get_most_probable_tokens_prefix(llama_model, prefix_embeds):

    if prefix_embeds.dim() == 2:
        prefix_embeds = prefix_embeds.unsqueeze(0)

    token_embeds = llama_model.lang_model.model.embed_tokens.weight
    token_embeds = token_embeds.to(device)

    prefix_norm = F.normalize(prefix_embeds, dim=-1)
    token_norm  = F.normalize(token_embeds, dim=-1)

    sims = torch.einsum("bph,vh->bpv", prefix_norm, token_norm)

    top1_sims, top1_ids = torch.topk(sims, k=1, dim=-1)
    top1_sims = top1_sims.tolist()

    # Decode
    tokens = llama_model.tokenizer.convert_ids_to_tokens( top1_ids[0, :, 0].tolist())

    return tokens, top1_sims


def main():
    transformation = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        ImageNorm()
        # transforms.Normalize([0.485, 0.456, 0.406],
        #                     [0.229, 0.224, 0.225])  # ImageNet stats
    ])

    clip_encoder = CLIP_DistilBert_ResNet().to(device)
    clip_encoder.load_state_dict(torch.load(CUSTOM_CLIP_FILE))
    clip_encoder.eval()

    llama_model = LlamaPrefix().to(device)
    llama_model.load_state_dict(torch.load(LANG_PREFIX_CHECKPOINT))
    llama_model.eval()

    # Load the seismic volume
    seismic_volume = carregar_arquivo(DATA_VOLUME_PATH)
    output_file = open(OUTPUT_FILENAME, 'w')

    print("Generating report...")

    inline = 0
    # traverse the seismic volume across inlines, break each inline in
    # 64x64 patches and save the caption for each patch.
    for inline, image_inline in enumerate(tqdm(seismic_volume[::INLINE_SKIP])):

        img_inline_arr = (image_inline       - image_inline.min()
                     ) / (image_inline.max() - image_inline.min()
        )
        img_inline_arr = np.swapaxes(img_inline_arr, 0, 1)

        img_inline_arr = (img_inline_arr * 255).astype(np.uint8)
        img_inline_arr = np.stack([img_inline_arr]*3, axis=-1)

        for xline in range(0, img_inline_arr.shape[0], 64):
            for z in range(0, img_inline_arr.shape[1], 64):
                np_patch = img_inline_arr[xline:xline+64, z:z+64]

                pil_image = Image.fromarray(np_patch, 'RGB')
                patch = transformation(pil_image)
                patch = patch.unsqueeze(0).to(device)

                with torch.no_grad():
                    image_embeds = clip_encoder.encode_image(patch)

                    prefix_embeds = \
                        llama_model.get_prefix_embeds_from_img_embeds(image_embeds)[0]
                    prefix_embeds = prefix_embeds.unsqueeze(0)

                    bos_id = llama_model.tokenizer.bos_token_id
                    bos_embed = llama_model.lang_model.model.embed_tokens(
                        torch.tensor([[bos_id]], device=device)
                    )
                    inputs_embeds = torch.cat([bos_embed, prefix_embeds], dim=1)

                    #prefix_tokens, _ = get_most_probable_tokens_prefix(
                    #    llama_model, inputs_embeds)

                    # print("Correct caption:", caption_batch[0])
                    generated_text = llama_model.generate_text_from_embeds(inputs_embeds)
                    cosine_similarities = calc_cosine_similarities(
                        llama_model, generated_text)

                print(f'Inline {inline+1}. Crosslines {xline+1} to {xline+65}.',
                      f'Depth {z+1} to {z+65}', file=output_file)

                print(cut_text_after_last_period(generated_text), file=output_file)
                # print("Label:", label_batch[0])
                #print("Closest tokens to prefix embeddings:", " ".join(prefix_tokens[1:]))
                print(cosine_similarities, file=output_file)

                print(100 * '-', file=output_file)

                # Medir a distância de cosseno com as legendas de referência e escolher com base nisso

        inline += INLINE_SKIP
    output_file.close()


if __name__ == "__main__":
    main()
