import torch
import torch.nn as nn
from models.resnet import Resnet1D

class TemporalAutoregressiveEncoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None,
                 temporal_hidden_size=128,
                 max_seq_length=1000):
        super().__init__()
        
        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)

        self.base_encoder = nn.Sequential(*blocks)
        self.time_embedding = nn.Embedding(max_seq_length, width) 
        self.output_layer = nn.Conv1d(width, output_emb_width, 3, 1, 1) 

    def forward(self,x, time_indices=None):
        """
        x: [batch_size, input_emb_width, sequence_length]
        time_indices: [batch_size, sequence_length]
        """
        batch_size, channels, seq_len = x.shape

     
        features = self.base_encoder(x)  # [batch_size, width, seq_len]

        features = features.transpose(1, 2)  # [B, T', width]

        # Generate time indices to match the downsampled length T_enc
        B, T_enc, _ = features.shape
        new_time_indices = torch.arange(T_enc, device=features.device).unsqueeze(0).repeat(B, 1)  # [B, T']

        time_emb = self.time_embedding(new_time_indices)  # [B, T', width]
        features = features + time_emb
        features = features.transpose(1, 2)

        # Generate new time indices to match the downsampled length
        T_enc = features.shape[1]
        time_indices = torch.arange(T_enc, device=x.device).expand(batch_size, T_enc)
        time_emb = self.time_embedding(time_indices)  # [B, T_enc, C]
        
        output = self.output_layer(features)  # [batch_size, output_emb_width, seq_len]
        
        return output

class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)