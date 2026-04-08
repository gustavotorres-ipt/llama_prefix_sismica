import torch

MAP_NETWORK = 'mlp'
LANG_PREFIX_CHECKPOINT =f'checkpoints/prefix_ckpt_parihaka_f3_{MAP_NETWORK}_v2.pth'

CLIP_VISION_MODEL ='checkpoints/resnet18_parihaka_f3_encoder.pth'
CLIP_LANGUAGE_MODEL ='checkpoints/lang_ckpt_parihaka_f3.pt'
CUSTOM_CLIP_FILE ='checkpoints/clip_parihaka_f3.pth'

N_EPOCHS = 30
LEARNING_RATE = 2e-6 # 1e-5
BATCH_SIZE = 32

IMG_SIZE = 96

IMAGE_FOLDER_TRAIN = 'data/janelas_parihaka_f3_balanceado/training'
TEXT_FOLDER_TRAIN = 'data/legendas_parihaka_f3_balanceado/training'
 
IMAGE_FOLDER_VAL = 'data/janelas_parihaka_f3_balanceado/validation'
TEXT_FOLDER_VAL = 'data/legendas_parihaka_f3_balanceado/validation'

PROJECTION_SIZE = 512

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CHROMA_DB_FILE = 'embeddings_sismicos'
N_EMBEDS_BATCH = 1000  # How many files to save each time
MAX_SIZE_MEMORY = 5000
