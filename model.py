import torch
import torch.nn as nn
import math

class DomainTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=8, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=2048):
        super(DomainTransformer, self).__init__()
        self.d_model = d_model  # Add this line
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        output = self.transformer(src, tgt)
        return self.output_linear(output)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
