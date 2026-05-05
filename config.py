import torch

MAP_NETWORK = 'mlp'
#MAP_NETWORK = 'transformer'

LANG_PREFIX_CHECKPOINT =f'checkpoints/prefix_f3_pari_peno_{MAP_NETWORK}.pth'
CLIP_VISION_MODEL ='checkpoints/resnet18_f3_pari_peno_FAN_encoder.pth'
CLIP_LANGUAGE_MODEL ='checkpoints/mlm_f3_pari_peno_FAN.pt'
CUSTOM_CLIP_FILE ='checkpoints/clip_f3_pari_peno_FAN.pth'

N_EPOCHS = 30
LEARNING_RATE = 1e-5
BATCH_SIZE = 32

IMG_SIZE = 96

IMAGE_FOLDER_TRAIN = 'data/imagens_f3_pari_peno_CLIP/training'
TEXT_FOLDER_TRAIN = 'data/legendas_f3_pari_peno_CLIP/training'
 
IMAGE_FOLDER_VAL = 'data/imagens_f3_pari_peno_CLIP/validation'
TEXT_FOLDER_VAL = 'data/legendas_f3_pari_peno_CLIP/validation'

PROJECTION_SIZE = 512

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CHROMA_DB_FILE = 'embeddings_sismicos'
N_EMBEDS_BATCH = 1000  # How many files to save each time
MAX_SIZE_MEMORY = 5000
