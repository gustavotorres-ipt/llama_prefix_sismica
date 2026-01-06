import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LlamaPrefix(nn.Module):
    def __init__(self, model_name = "meta-llama/Llama-3.2-1B",
                 prefix_len=10,
                 # max_sent_length=128,
                 clip_hidden_dim=512) -> None:
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.lang_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        for p in self.lang_model.parameters():
            p.requires_grad = False

        self.hidden_size = self.lang_model.config.hidden_size
        self.prefix_len = prefix_len

        # MLP using to project the dimensions of the CLIP output
        # to (llama latent space X number of prefix embeddings).
        self.proj_mlp = nn.Sequential(
            nn.Linear(clip_hidden_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size*self.prefix_len),
            nn.LayerNorm(self.hidden_size * self.prefix_len)
        )

    def get_attention_mask(self, input_ids):
        batch_size = input_ids.shape[0]

        prefix_mask = torch.ones(
            batch_size,
            self.prefix_len,
            device=device
        )

        text_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return torch.cat([prefix_mask, text_mask], dim=1)

    def prefix_and_text_to_logits(self, image_embeds, input_ids):
        x = self.proj_mlp(image_embeds)

        prefix_embeds = x.view(x.size(0), self.prefix_len, self.hidden_size)
        text_embeds = self.lang_model.model.embed_tokens(input_ids)
        input_embeds = torch.cat( [prefix_embeds, text_embeds], dim=1 )

        output = self.generate_output_from_embeds(input_embeds, input_ids)
        return output

    def generate_output_from_embeds(self, inputs_embeds, input_ids):
        attention_mask = self.get_attention_mask(input_ids)

        # outputs = self.lang_model.generate(
        #     inputs_embeds=input_embeds,
        #     attention_mask=attention_mask,
        #     max_new_tokens=128,
        #     #temperature=0.1,
        #     top_p=0.9,
        #     do_sample=True,
        #     pad_token_id=self.tokenizer.eos_token_id,
        #     return_dict_in_generate=True,
        #     output_scores=True,
        # )
        outputs = self.lang_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )

        # output_text = self.tokenizer.decode(
        #     output_ids[0],
        #     skip_special_tokens=True
        # )
        return outputs

    def tokenize_texts(self, texts):
        inputs = self.tokenizer(
            texts, truncation=True, padding=True,
            return_tensors="pt", max_length=128,
        ).to(device)
        return inputs


    def forward(self, image_embeds, input_ids):
        x = self.proj_mlp(image_embeds)

        prefix_embeds = x.view(x.size(0), self.prefix_len, self.hidden_size)
        text_embeds = self.lang_model.model.embed_tokens(input_ids)
        input_embeds = torch.cat( [prefix_embeds, text_embeds], dim=1 )

        output = self.generate_output_from_embeds(input_embeds, input_ids)
        return output

    def generate_text_for_prompt(self, prompts):
        inputs = self.tokenize_texts(prompts)

        with torch.no_grad():
            generated_content = self.lang_model.generate(
                **inputs,
                max_new_tokens=128,
                pad_token_id=self.tokenizer.eos_token_id
            )[0]
            return self.tokenizer.decode(
                generated_content, skip_special_tokens=True)

