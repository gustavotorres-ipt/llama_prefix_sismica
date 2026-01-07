import torch
import copy
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.llama_prefix import LlamaPrefix
from models.clip_model import CLIP_DistilBert_ResNet
from config import LEARNING_RATE, CUSTOM_CLIP_FILE, N_EPOCHS, BATCH_SIZE, device, LANG_PREFIX_CHECKPOINT
from dataset import load_datasets

torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.benchmark = False


def main():
    _, val_dataset = load_datasets()

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    clip_encoder = CLIP_DistilBert_ResNet().to(device)
    clip_encoder.load_state_dict(torch.load(CUSTOM_CLIP_FILE))

    llama_model = LlamaPrefix().to(device)
    llama_model.load_state_dict(torch.load(LANG_PREFIX_CHECKPOINT))


    for image_batch, caption_batch in tqdm(val_loader):
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
        print("Correct caption:", caption_batch[0])
        print("Generated:", llama_model.generate_text_from_embeds(inputs_embeds))
        print(100 * '-')


if __name__ == "__main__":
    main()
