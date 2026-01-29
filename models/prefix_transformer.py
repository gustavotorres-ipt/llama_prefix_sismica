import torch
from torch import nn

class PrefixTransformer(nn.Module):
    def __init__(
        self,
        clip_dim,
        hidden_size,
        prefix_len,
        n_layers=4,
        n_heads=8,
    ):
        super().__init__()

        self.prefix_len = prefix_len
        self.hidden_size = hidden_size

        self.input_proj = nn.Linear(clip_dim, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.prefix_tokens = nn.Parameter(
            torch.randn(prefix_len, hidden_size)
        )

    def forward(self, image_embeds):
        B = image_embeds.size(0)

        x = self.input_proj(image_embeds).unsqueeze(1)   # (B, 1, H)

        prefix = self.prefix_tokens.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([x, prefix], dim=1)                # (B, 1+P, H)

        x = self.transformer(x)

        return x[:, 1:, :]                                # (B, P, H)
