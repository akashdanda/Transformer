import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path

#tokenizer builds the vocab of numbers, maps words to numbers
#can create custom tokens: (Ex: Padding, Start, End)

#get's the sentences in specific language
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang)) # create file path for saving and loading tokenizer
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) #split words based on word not subword (unk for words not seen before)
        tokenizer.pre_tokenizer = Whitespace() # preliminary splitting step
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency = 2) #only words seen at least twice get added to training data
        #UNK, PAD, SOS, and EOS are special: padding, start of sentence, end of sentence, and unknown
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), training = trainer)#trains tokenizer by learning from vocab in dataset
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer