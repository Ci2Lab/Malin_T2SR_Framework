import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.nn.functional as T
from torch.utils.data import Dataset, random_split, DataLoader
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
from torch.utils.data import DataLoader, TensorDataset
from math import pi, sin, cos
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import joblib

"""
IMPLEMENTATION OF T2SR FRAMEWORK

"""


class T2SR(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dropout_p):
        super(T2SR, self).__init__()
        
        # Embeds the input sequence to a higher dimensional space
        self.input_embedding = nn.Linear(input_size, d_model)
        
        # Adds positional encodings to provide temporal information
        self.pos_encoder = PositionalEncoding(d_model, dropout_p)

        # Encoder layer definition using the Transformer's encoder layer
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout_p)
        
        # Transformer encoder processes the input sequence
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_encoder_layers)

        # Decoder layer definition similar to the encoder but for decoding
        self.decoder_layers = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout_p)
        
        # Transformer decoder uses encoder's outputs and processes target sequence
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layers, num_layers=num_decoder_layers)

        # Final linear layer that projects the decoder output to the desired output size
        self.output_layer = nn.Linear(d_model, output_size)
        
        

    def forward(self, src, tgt):
        
        # Process the source sequence through the input embedding and positional encoding
        src = self.input_embedding(src)
        src = self.pos_encoder(src)
        
        # Encoder generates a memory for the source sequence
        memory = self.transformer_encoder(src)

        # Process the target sequence for decoding
        batch_size, seq_len = tgt.size()
        tgt = tgt.view(-1, 1)  # Flatten target for embedding
        tgt = self.input_embedding(tgt)  # Embed target sequence
        tgt = tgt.view(batch_size, seq_len, -1)  # Reshape back to batch format
        tgt = tgt.permute(1, 0, 2)  # Permute for the transformer decoder input format
        tgt = self.pos_encoder(tgt)  # Add positional encoding

        # Decoder generates the output sequence using the memory from the encoder
        output = self.transformer_decoder(tgt, memory)
        output = output.permute(1, 0, 2)  # Permute back to original format
        output = output[:, -1, :].unsqueeze(1)  # Select the last output for each sequence
        output = self.output_layer(output)  # Pass through the output linear layer
        output = output.permute(1, 0, 2)  # Optional: Adjust dimensions if needed

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        
        # Compute the positional encodings once in log space
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices in the tensor
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        self.register_buffer('pe', pe)  # Register pe as a buffer that is not a model parameter
        
        

    def forward(self, x):
        
        # Add positional encoding to each position in the input sequence
        x = x + self.pe[:x.size(0)]
        
        return self.dropout(x)  # Apply dropout for regularization


