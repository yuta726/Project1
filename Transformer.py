import torch
from Decoder import TransformerDecoder
from Encoder import TransformerEncoder
from torch import nn
from embedding import TokenEmbedding
from PositionalEncoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, max_len:int = 60, d_model:int = 512, heads_num:int = 8, d_ff:int = 2048, N:int = 6, 
                 dropout_rate:float = 0.1, layer_norm_eps:float = 1e-5, 
                 device:torch.device = torch.device("cpu")) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.heads_num = heads_num
        self.d_ff = d_ff
        self.N = N
        self.dropout_rate = dropout_rate
        self.layer_norm_eps = layer_norm_eps
        self.device = device

        self.token_embedding_src = TokenEmbedding(1, d_model)
        self.token_embedding_tgt = TokenEmbedding(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout_rate, max_len)

        self.encoder = TransformerEncoder(max_len, d_model, N, d_ff, heads_num, dropout_rate, layer_norm_eps)
        self.decoder = TransformerDecoder(max_len, d_model, N, d_ff, heads_num, dropout_rate, layer_norm_eps)
        
        self.linear = nn.Linear(d_model, 1)

    def forward(self, src:torch.Tensor, tgt:torch.Tensor) -> torch.Tensor:
        #mask
        src = self.positional_encoding(self.token_embedding_src(src))
        mask_src = self._generate_square_subsequent_mask(src.shape[1], src.shape[0])
        src = self.encoder(src, mask_src)

        tgt = self.positional_encoding(self.token_embedding_tgt(tgt))
        mask_tgt = self._generate_square_subsequent_mask(tgt.shape[1], tgt.shape[0])
        dec_output = self.decoder(tgt, src, None, mask_tgt)

        return self.linear(dec_output)


    def _generate_square_subsequent_mask(self, seq_len, batch_size):
        mask = torch.triu(torch.full((seq_len, seq_len), 1), diagonal=1)
        mask = torch.where(mask==1, True, False)
        mask = mask.repeat(batch_size, 1, 1)
        return mask.to(self.device)
    
    def encode(self, src, mask_src):
        return self.encoder(self.positional_encoding(self.token_embedding_src(src)), mask_src)
    
    def decode(self, tgt, src, mask_src, mask_tgt):
        return self.decoder(self.positional_encoding(self.token_embedding_tgt(tgt)), src, mask_src, mask_tgt)
