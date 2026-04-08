import torch
from PIL import Image
from torchvision import transforms
from models.clip_model import CLIP_DistilBert_ResNet
from models.llama_prefix import LlamaPrefix
from config import device, CUSTOM_CLIP_FILE, LANG_PREFIX_CHECKPOINT, IMG_SIZE
from dataset import ImageNorm


class CaptionGenerator:
    def __init__(self):
        # Load img transformation pipeline
        self.encoder = CLIP_DistilBert_ResNet() 
        self.decoder = LlamaPrefix()

        self.encoder.load_state_dict(torch.load(CUSTOM_CLIP_FILE))
        self.decoder.load_state_dict(torch.load(LANG_PREFIX_CHECKPOINT))

        self.transformation = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            ImageNorm(),
        ])

    def generate_caption(self, input_image: str):
        # Load image
        image = Image.open(input_image).convert("RGB")

        # Transform image
        tensor_image = self.transformation(image)
        tensor_image = tensor_image.unsqueeze(0)

        # Encode image
        image_embeds = self.encoder.encode_image(tensor_image)

        # Get preffix embeddings
        prefix_embeds = \
            self.decoder.get_prefix_embeds_from_img_embeds(image_embeds)[0]
        
        prefix_embeds = prefix_embeds.unsqueeze(0).to(device)
        bos_id = self.decoder.tokenizer.bos_token_id
        bos_embed = self.decoder.lang_model.model.embed_tokens(
            torch.tensor([[bos_id]], device=device)
        )
        inputs_embeds = torch.cat([bos_embed, prefix_embeds], dim=1)

        # Generate caption from image embeds
        caption = self.decoder.generate_text_from_embeds(inputs_embeds)

        if caption[-1] != '.':
            caption = '.'.join(caption.split('.')[:-1])

        return caption
