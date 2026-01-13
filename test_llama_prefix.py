import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.models.oneformer.modeling_oneformer import PredictionBlock
from models.llama_prefix import LlamaPrefix
from models.clip_model import CLIP_DistilBert_ResNet
from config import CUSTOM_CLIP_FILE, device, LANG_PREFIX_CHECKPOINT
from dataset import load_datasets

torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.benchmark = False

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
    _, val_dataset = load_datasets()

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    with torch.no_grad():
        clip_encoder = CLIP_DistilBert_ResNet().to(device)
        clip_encoder.load_state_dict(torch.load(CUSTOM_CLIP_FILE))
        clip_encoder.eval()

        llama_model = LlamaPrefix().to(device)
        llama_model.load_state_dict(torch.load(LANG_PREFIX_CHECKPOINT))
        llama_model.eval()


        for image_batch, caption_batch, label_batch in tqdm(val_loader):
            image_batch = image_batch.to(device)
            image_embeds = clip_encoder.encode_image(image_batch)

            prefix_embeds = \
                llama_model.get_prefix_embeds_from_img_embeds(image_embeds)[0]
            prefix_embeds = prefix_embeds.unsqueeze(0)

            bos_id = llama_model.tokenizer.bos_token_id
            bos_embed = llama_model.lang_model.model.embed_tokens(
                torch.tensor([[bos_id]], device=device)
            )
            inputs_embeds = torch.cat([bos_embed, prefix_embeds], dim=1)

            print(100 * '-')
            #prefix_tokens, _ = get_most_probable_tokens_prefix(
            #    llama_model, inputs_embeds)

            print("Correct caption:", caption_batch[0])
            print("Generated:", llama_model.generate_text_from_embeds(inputs_embeds))
            print("Label:", label_batch[0])
            #print("Closest tokens to prefix embeddings:", " ".join(prefix_tokens[1:]))
            print(100 * '-')


if __name__ == "__main__":
    main()
