import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import device, MAP_NETWORK
from models.prefix_transformer import PrefixTransformer


class LlamaPrefix(nn.Module):
    def __init__(self, model_name = "meta-llama/Llama-3.2-1B",
                 prefix_len=13,
                 max_new_tokens=22,
                 clip_hidden_dim=512) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.lang_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )

        self.hidden_size = self.lang_model.config.hidden_size
        self.prefix_len = prefix_len
        self.max_new_tokens = max_new_tokens

        # MLP using to project the dimensions of the CLIP output
        # to (llama latent space X number of prefix embeddings).
        if MAP_NETWORK.lower() == 'mlp':
            self.proj_mlp = nn.Sequential(
                nn.Linear(clip_hidden_dim, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.hidden_size*2),
                nn.GELU(),
                nn.Linear(self.hidden_size*2, self.hidden_size*self.prefix_len),
                nn.LayerNorm(self.hidden_size * self.prefix_len)
            )
        else:
            self.proj_mlp = PrefixTransformer(
                clip_dim=clip_hidden_dim,
                hidden_size=self.hidden_size,
                prefix_len=self.prefix_len,
            )
        for p in self.lang_model.parameters():
            p.requires_grad = False

    def get_attention_mask(self, input_ids):
        batch_size = input_ids.shape[0]

        prefix_mask = torch.ones(
            batch_size,
            self.prefix_len,
            device=device
        ).long()
        text_mask = (input_ids != self.tokenizer.pad_token_id)

        return torch.cat([prefix_mask, text_mask], dim=1).long().to(device)

    def get_prefix_embeds_from_img_embeds(self, image_embeds):
        x = self.proj_mlp(image_embeds).to(dtype=self.lang_model.dtype)

        prefix_embeds = x.view(x.size(0), self.prefix_len, self.hidden_size)
        return prefix_embeds

    def prefix_and_text_to_logits(self, image_embeds, input_ids):
        prefix_embeds = self.get_prefix_embeds_from_img_embeds(image_embeds)
        text_embeds = self.lang_model.model.embed_tokens(input_ids)
        input_embeds = torch.cat( [prefix_embeds, text_embeds], dim=1 )

        output = self.get_output_for_embeds(input_embeds, input_ids)
        return output

    def get_output_for_embeds(self, inputs_embeds, input_ids):
        attention_mask = self.get_attention_mask(input_ids)

        outputs = self.lang_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        return outputs

    def tokenize_texts(self, texts):
        inputs = self.tokenizer(
            texts, truncation=True, padding=True,
            return_tensors="pt", max_length=self.max_new_tokens,
        ).to(device)
        return inputs


    def forward(self, image_embeds, input_ids):
        x = self.proj_mlp(image_embeds).to(dtype=self.lang_model.dtype)

        prefix_embeds = x.view(x.size(0), self.prefix_len, self.hidden_size)
        text_embeds = self.lang_model.model.embed_tokens(input_ids)

        input_embeds = torch.cat( [prefix_embeds, text_embeds], dim=1 )

        output = self.get_output_for_embeds(input_embeds, input_ids)
        return output

    def generate_text_for_prompt(self, prompts):
        inputs = self.tokenize_texts(prompts)

        with torch.no_grad():
            generated_content = self.lang_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id
            )[0]
            return self.tokenizer.decode(
                generated_content, skip_special_tokens=True)

    def generate_text_from_embeds(self, inputs_embeds):
        max_new_tokens = self.max_new_tokens - \
            inputs_embeds.size(1) + self.prefix_len

        attention_mask = torch.ones(
            inputs_embeds.size(0),
            inputs_embeds.size(1),
            device=device
        ).long()

        generated_ids = self.lang_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_p=0.85,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # early_stopping=True,
        )[0]

        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=True)
