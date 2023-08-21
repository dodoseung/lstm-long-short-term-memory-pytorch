import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchtext.data import Field, BucketIterator

import re
from utils import save_model, load_yaml

# Set the configuration
config = load_yaml("./config/vit_cifar10_config.yml")

# Training setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(config['data']['seed'])
if device == 'cuda':
  torch.cuda.manual_seed_all(config['data']['seed'])

# Data loading
# Save https://www.gutenberg.org/cache/epub/21/pg21.txt as aesop.txt
file_path = config['data']['path']
with open(file_path, encoding='utf-8-sig') as f:
    text = f.read()

# Text preprocessing
text = text.lower()
text = '|' + text
text = text.replace('\n\n\n\n\n', '|')
text = text.replace('\n', ' ')
text = text.replace('..', '.')
text = re.sub('([!"#$%&()*+,-./:;<=>?@[\]^_`{|}~])', r' \1 ', text)
text = re.sub('\s{2,}', ' ', text)

# Tokenize
field = Field(tokenize = 'spacy', tokenizer_language='en', lower=True, batch_first=True)
field.build_vocab(text, min_freq=2)