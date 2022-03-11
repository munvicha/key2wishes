import transformers
import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 256
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
EPOCHS = 10
MODEL_NAME = 'google/mt5-small'
MODEL_PATH = r'E:\K2W\model'
TRAINING_FILE = r'E:\K2W\data\k2w.csv'
TOKENIZER = transformers.T5Tokenizer.from_pretrained(MODEL_NAME)