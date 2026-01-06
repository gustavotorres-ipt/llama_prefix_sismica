import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.llama_prefix import LlamaPrefix
from models.clip_model import CLIP_DistilBert_ResNet
from config import LEARNING_RATE, CUSTOM_CLIP_FILE, N_EPOCHS, BATCH_SIZE
from dataset import load_datasets

torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_val_loss(val_loader, llama_model, clip_encoder, criterion):
    total_val_loss = 0

    with torch.no_grad():
        batch = 0

        for (image_batch, text_batch) in tqdm(val_loader):
            image_batch = image_batch.to(device)

            text_inputs = llama_model.tokenize_texts(text_batch)
            inputs_ids = text_inputs.input_ids

            image_embeds = clip_encoder.encode_image(image_batch)

            outputs = llama_model( image_embeds, inputs_ids )

            prefix_len = llama_model.prefix_len
            pred_logits = outputs.logits[:, prefix_len:-1,:]

            target_ids = inputs_ids[:, 1:]

            pred_ids = torch.argmax(pred_logits, dim=-1)

            if batch % 20 == 0:
                print("Correct:", llama_model.tokenizer.decode( target_ids[0], skip_special_tokens=True))
                print("Predicted:", llama_model.tokenizer.decode( pred_ids[0], skip_special_tokens=True))
                print("-" * 100)

            loss = criterion(
                pred_logits.reshape(-1, pred_logits.size(-1)), target_ids.reshape(-1),
            )
            num_tokens = (target_ids != llama_model.tokenizer.pad_token_id).sum()
            loss = loss / num_tokens
            total_val_loss += loss.item()

            batch += 1

    return total_val_loss / len(val_loader)

def calc_train_loss(train_loader, llama_model, clip_encoder, criterion, optimizer):
    total_train_loss = 0

    for (image_batch, text_batch) in tqdm(train_loader):
        image_batch = image_batch.to(device)

        text_inputs = llama_model.tokenize_texts(text_batch)
        inputs_ids = text_inputs.input_ids

        image_embeds = clip_encoder.encode_image(image_batch)

        outputs = llama_model( image_embeds, inputs_ids )

        prefix_len = llama_model.prefix_len
        pred_logits = outputs.logits[:, prefix_len:-1,:]

        target_ids = inputs_ids[:, 1:]

        pred_ids = torch.argmax(pred_logits, dim=-1)
        # if idx % 40 == 0:
        # print("Correct:", llama_model.tokenizer.decode( target_ids[0], skip_special_tokens=True))
        # print("Predicted:", llama_model.tokenizer.decode( pred_ids[0], skip_special_tokens=True))

        loss = criterion(
            pred_logits.reshape(-1, pred_logits.size(-1)), target_ids.reshape(-1),
        )
        num_tokens = (target_ids != llama_model.tokenizer.pad_token_id).sum()
        loss = loss / num_tokens
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_train_loss / len(train_loader)
    # return total_loss / len(train_loader)

def main():
    train_dataset, val_dataset = load_datasets()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    clip_encoder = CLIP_DistilBert_ResNet().to(device)
    llama_model = LlamaPrefix().to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=llama_model.tokenizer.pad_token_id, reduction="sum"
    )
    optimizer = torch.optim.Adam(
        llama_model.proj_mlp.parameters(), lr=LEARNING_RATE
    )
    clip_encoder.load_state_dict(torch.load(CUSTOM_CLIP_FILE))

    for epoch in range(1, N_EPOCHS+1):
        avg_train_loss = calc_train_loss(train_loader, llama_model, clip_encoder, criterion, optimizer)
        avg_val_loss = calc_val_loss(val_loader, llama_model, clip_encoder, criterion)
            # idx += 1

        print(f"Epoch {epoch} - Train loss: {avg_train_loss}. Val loss: {avg_val_loss}")

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
