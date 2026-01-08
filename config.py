import torch

LANG_PREFIX_CHECKPOINT = 'checkpoints/lang_prefix_ckpt.pth'

CLIP_VISION_MODEL = 'checkpoints/resnet18_image_encoder.pth'
CLIP_LANGUAGE_MODEL = 'checkpoints/seismic_distilbert.pt'
CUSTOM_CLIP_FILE = 'checkpoints/clip_sismico_sintetico.pth'

N_EPOCHS = 15
LEARNING_RATE = 1e-5

IMAGE_FOLDER_TRAIN = 'data/images/training'
TEXT_FOLDER_TRAIN = 'data/captions/training'
IMAGE_FOLDER_VAL = 'data/images/validation'
TEXT_FOLDER_VAL = 'data/captions/validation'

PROJECTION_SIZE = 512
BATCH_SIZE = 16

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CHROMA_DB_FILE = 'embeddings_sismicos'
N_EMBEDS_BATCH = 1000  # How many files to save each time


# CLIP_VISION_MODEL = 'checkpoints/resnet18_inline_xline_todos_tams_balanceado_encoder.pth'
# CLIP_LANGUAGE_MODEL = 'checkpoints/mlm_sismofacies.pt'
# CUSTOM_CLIP_FILE = 'checkpoints/clip_janelas_seismic_faces_todos_tamanhos.pth'
# 
# N_EPOCHS = 40
# LEARNING_RATE = 1e-5
# WEIGHTS_PATH = 'checkpoints/decoder_seismic_facies_todos_tams.pth'
# 
# IMAGE_FOLDER_TRAIN = 'data/janelas_iline_xline_32_40_64_balanceado/training'
# TEXT_FOLDER_TRAIN = 'data/legendas_iline_xline_32_40_64_balanceado/training'
# IMAGE_FOLDER_VAL = 'data/janelas_iline_xline_32_40_64_balanceado/validation'
# TEXT_FOLDER_VAL = 'data/legendas_iline_xline_32_40_64_balanceado/validation'
# 
# PROJECTION_SIZE = 128
# 
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 
# CHROMA_DB_FILE = 'embeddings_sismicos'
# N_EMBEDS_BATCH = 1000  # How many files to save each time
# MAX_SIZE_MEMORY = 5000

# CLIP_VISION_MODEL = 'checkpoints/resnet34_inline_xline_64_bal_encoder.pth'
# CLIP_LANGUAGE_MODEL = 'checkpoints/mlm_sismofacies.pt'
# CUSTOM_CLIP_FILE = 'checkpoints/clip_janelas_64_il_xl_bal.pth'
# 
# N_EPOCHS = 40
# LEARNING_RATE = 1e-5
# WEIGHTS_PATH = 'checkpoints/decoder_janela_64_il_xl_bal.pth'
# 
# IMAGE_FOLDER_TRAIN = 'data/janelas_inline_xline_64_balanceado/training'
# TEXT_FOLDER_TRAIN = 'data/legendas_inline_xline_64_balanceado/training'
# IMAGE_FOLDER_VAL = 'data/janelas_inline_xline_64_balanceado/validation'
# TEXT_FOLDER_VAL = 'data/legendas_inline_xline_64_balanceado/validation'
# 
# PROJECTION_SIZE = 512
# 
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 
# CHROMA_DB_FILE = 'embeddings_sismicos'
# N_EMBEDS_BATCH = 1000  # How many files to save each time
# MAX_SIZE_MEMORY = 5000
