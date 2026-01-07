import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torchvision import models
from config import CLIP_VISION_MODEL, CLIP_LANGUAGE_MODEL, PROJECTION_SIZE, device

class CLIP_DistilBert_ResNet(nn.Module):
    def __init__(
        self,
        text_model_name: str = 'distilbert-base-uncased',
        embed_dim: int = PROJECTION_SIZE,
        image_pretrained: bool = True,
        learnable_temp = 1.
    ) -> None:
        super().__init__()

        # -----------------------------
        # 1. TEXT ENCODER (DistilBERT)
        # -----------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            text_model_name, local_files_only=True)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.text_encoder = AutoModel.from_pretrained(CLIP_LANGUAGE_MODEL)
        # self.text_encoder = DistilBertModel.from_pretrained(text_model_name)

        text_hidden_dim = self.text_encoder.config.dim  # usually 768

        # Text projection into shared space
        self.embed_dim = embed_dim
        self.text_proj = nn.Linear(text_hidden_dim, embed_dim)

        # -----------------------------
        # 2. IMAGE ENCODER (ResNet-34)
        # -----------------------------
        if "resnet34" in CLIP_VISION_MODEL:
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            model = nn.Sequential(*list(model.children())[:-1])

        else: # resnet18
            model = models.resnet18(weights=None)
            model = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
                model.avgpool
            ) 
        model.load_state_dict(torch.load(CLIP_VISION_MODEL))
        self.image_encoder = model
        
        image_hidden_dim = 512  # ResNet-18 output dimension

        # Image projection into shared space
        self.image_proj = nn.Linear(image_hidden_dim, embed_dim)

        # Temperature parameter (learnable)
        self.logit_scale = nn.Parameter(
            torch.ones([]) * (1 / learnable_temp)
        )

        for p in self.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Encode text using DistilBERT → pooled embedding → projection → norm
    # ------------------------------------------------------------------
    def encode_text(self, texts):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        # move to same device as module params
        tokens = {k: v.to(device) for k, v in tokens.items()}

        outputs = self.text_encoder(**tokens)

        # Use CLS token embedding (DistilBERT has CLS at index 0)
        text_emb = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_dim)

        text_emb = self.text_proj(text_emb)  # project
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)  # L2 normalize
        return text_emb

    # --------------------------------------------------------------
    # Encode image using ResNet-18 → projection → norm
    # --------------------------------------------------------------
    def encode_image(self, images):
        img_feat = self.image_encoder(images)  # (batch, 512)
        # img_emb = self.image_proj(img_feat[:, :, 0, 0])
        img_emb = img_feat.squeeze(-1).squeeze(-1)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        img_emb = self.image_proj(img_emb)
        return img_emb

    # --------------------------------------------------------------
    # Forward pass: compute contrastive logits
    # --------------------------------------------------------------
    def forward(self, images, texts):
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Compute cosine similarity
        logits_per_image = image_features @ text_features.T
        logits_per_text = text_features @ image_features.T

        # Use learned temperature
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image *= logit_scale
        logits_per_text *= logit_scale

        return logits_per_image, logits_per_text


def load_custom_encoders():
    # image_encoder
    # resnet18 = models.resnet18(pretrained=False)
    # image_encoder = nn.Sequential(
    #     resnet18.conv1,
    #     resnet18.bn1,
    #     resnet18.relu,
    #     resnet18.maxpool,
    #     resnet18.layer1,
    #     resnet18.layer2,
    #     resnet18.layer3,
    #     resnet18.layer4,
    #     resnet18.avgpool
    # )
    if "resnet50" in CLIP_VISION_MODEL:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        image_encoder = nn.Sequential(*list(model.children())[:-1])
    elif "resnet18" in CLIP_VISION_MODEL:
        resnet18 = models.resnet18(weights=None)
        model = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4,
            resnet18.avgpool
        ) 
        image_encoder = model
    else:
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        image_encoder = nn.Sequential(*list(model.children())[:-1])

    text_encoder = AutoModel.from_pretrained(CLIP_LANGUAGE_MODEL)

    image_encoder.load_state_dict(torch.load(CLIP_VISION_MODEL))  # Custom image encoder
    image_encoder.eval()

    return image_encoder, text_encoder
