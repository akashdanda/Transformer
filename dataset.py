from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BillingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype = torch.int64)
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]#source text
        tgt_text = src_target_pair['translation'][self.tgt_lang]#target text

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids #will tokenize sentence
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids #will tokenize target sentence

        enc_input_tokens = enc_input_tokens[: self.seq_len - 2]
        dec_input_tokens = dec_input_tokens[: self.seq_len - 1]

        enc_num_padding_tokens = self.seq_len - (len(enc_input_tokens) + 2)  # +2 for SOS, EOS
        dec_num_padding_tokens = self.seq_len - (len(dec_input_tokens) + 1)  # +1 for SOS or EOS

        #make sure that the length of seq not too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence too long")
        

        encoder_input = torch.cat( #concatenates 
            [
                self.sos_token, #start of sentence
                torch.tensor(enc_input_tokens, dtype=torch.int64), #encoder input tokens
                self.eos_token, #end of sentence
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64), #fill rest with padding
            ],
            dim=0,
        )

 
        decoder_input = torch.cat( #concatenates
            [
                self.sos_token, #start of sentence
                torch.tensor(dec_input_tokens, dtype=torch.int64), #decoder input tokens
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64), #fill the rest with padding
            ],
            dim=0,
        )


        #label is ground truth object that model is supposed to predict
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Double check the size of the tensors are seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
    
#see values past diagonal & make sure diag values are 0
def casual_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0