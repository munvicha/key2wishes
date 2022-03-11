import config
from transformers import T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained(config.MODEL_PATH)