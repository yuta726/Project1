import torch
from torch import nn
from torch.nn import LayerNorm

from FFN import FFN
from MultiHeadAttention import MultiHeadAttention

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model:int, d_ff:int, heads_num:int, 
                 dropout_rate:float, layer_norm_eps:float) -> None:
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(d_model, heads_num)
        self.dropout_self_attention = nn.Dropout(dropout_rate)
        self.layer_norm_self_attention = LayerNorm(d_model, eps=layer_norm_eps)

        self.ffn = FFN(d_model, d_ff)
        self.dropout_ffn = nn.Dropout(dropout_rate)
        self.layer_norm_ffn = LayerNorm(d_model, eps=layer_norm_eps)


    def forward(self, x:torch.Tensor, mask:torch.Tensor = None) -> torch.Tensor:
        x = self.layer_norm_self_attention(self.__self_attention_block(x, mask) + x)
        x = self.layer_norm_ffn(self.__feed_forward_block(x) + x)
        return x
    
    def __self_attention_block(self, x:torch.Tensor, mask:torch.Tensor)->torch.Tensor:
        x = self.multi_head_attention(x, x, x, mask)
        return self.dropout_self_attention(x)
    
    def __feed_forward_block(self, x:torch.Tensor) -> torch.Tensor:
        return self.dropout_ffn(self.ffn(x))
    
class TransformerEncoder(nn.Module):
    def __init__(self, max_len:int, d_model:int,
                 N:int, d_ff:int, heads_num:int, dropout_rate:float,
                 layer_norm_eps:float) -> None:
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model, d_ff, heads_num, dropout_rate, layer_norm_eps
                )
                for _ in range(N)
            ]
        )

    def forward(self, x:torch.Tensor, mask:torch.Tensor = None) -> torch.Tensor:
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return x