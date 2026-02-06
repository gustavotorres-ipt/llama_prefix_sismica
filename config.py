import torch

# LANG_PREFIX_CHECKPOINT ='checkpoints/prefix_ckpt_parihaka_f3.pth'

# CLIP_VISION_MODEL = 'checkpoints/resnet18_image_encoder.pth'
# CLIP_LANGUAGE_MODEL = 'checkpoints/seismic_distilbert.pt'
# CUSTOM_CLIP_FILE = 'checkpoints/clip_sismico_sintetico.pth'
# 
# N_EPOCHS = 15
# LEARNING_RATE = 1e-5
# BATCH_SIZE = 8
# 
# IMAGE_FOLDER_TRAIN = 'data/images/training'
# TEXT_FOLDER_TRAIN = 'data/captions/training'
# IMAGE_FOLDER_VAL = 'data/images/validation'
# TEXT_FOLDER_VAL = 'data/captions/validation'
# 
# PROJECTION_SIZE = 512
# 
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 
# CHROMA_DB_FILE = 'embeddings_sismicos'
# N_EMBEDS_BATCH = 1000  # How many files to save each time


# CLIP_VISION_MODEL = 'checkpoints/resnet18_inline_xline_todos_tams_balanceado_encoder.pth'
# CLIP_LANGUAGE_MODEL = 'checkpoints/mlm_sismofacies.pt'
# CUSTOM_CLIP_FILE = 'checkpoints/clip_janelas_seismic_faces_todos_tamanhos.pth'
# 
# N_EPOCHS = 40
# LEARNING_RATE = 1e-5
# BATCH_SIZE = 8
# 
# IMAGE_FOLDER_TRAIN = 'data/janelas_iline_xline_32_40_64_balanceado/training'
# TEXT_FOLDER_TRAIN = 'data/legendas_iline_xline_32_40_64_balanceado/training'
# IMAGE_FOLDER_VAL = 'data/janelas_iline_xline_32_40_64_balanceado/validation'
# TEXT_FOLDER_VAL = 'data/legendas_iline_xline_32_40_64_balanceado/validation'
# 
# PROJECTION_SIZE = 512
# 
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 
# CHROMA_DB_FILE = 'embeddings_sismicos'
# N_EMBEDS_BATCH = 1000  # How many files to save each time
# MAX_SIZE_MEMORY = 5000

MAP_NETWORK = 'transformer'
LANG_PREFIX_CHECKPOINT =f'checkpoints/prefix_ckpt_parihaka_f3_{MAP_NETWORK}.pth'

CLIP_VISION_MODEL = 'checkpoints/resnet18_parihaka_f3_encoder.pth'
CLIP_LANGUAGE_MODEL = 'checkpoints/lang_ckpt_parihaka_f3.pt'
CUSTOM_CLIP_FILE = 'checkpoints/clip_parihaka_f3.pth'

N_EPOCHS = 30
LEARNING_RATE = 2e-6 # 1e-5
BATCH_SIZE = 32

IMAGE_FOLDER_TRAIN = 'data/janelas_parihaka_f3_balanceado/training'
TEXT_FOLDER_TRAIN = 'data/legendas_parihaka_f3_balanceado/training'
 
IMAGE_FOLDER_VAL = 'data/janelas_parihaka_f3_balanceado/validation'
TEXT_FOLDER_VAL = 'data/legendas_parihaka_f3_balanceado/validation'

PROJECTION_SIZE = 512

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CHROMA_DB_FILE = 'embeddings_sismicos'
N_EMBEDS_BATCH = 1000  # How many files to save each time
MAX_SIZE_MEMORY = 5000
