import math
import torch
import torch.nn as nn
from typing import List, Dict, Tuple


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class CTCTransformer(nn.Module):
    def __init__(self, input_dim, vocab_size, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, max_len=5000, use_cnn=True):
        super().__init__()
        self.d_model = d_model
        self.use_cnn = use_cnn

        if use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(input_dim),
                nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),
                nn.ReLU()
            )

        self.input_fc = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.output_fc = nn.Linear(d_model, vocab_size)

    def forward(self, features, src_key_padding_mask=None):
        if self.use_cnn:
            features = features.permute(0, 2, 1)
            features = self.cnn(features)
            features = features.permute(0, 2, 1)

        x = self.input_fc(features)
        x = self.input_dropout(x)
        x = self.positional_encoding(x)

        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.output_fc(encoded)  # (B, T, vocab_size)
        return logits

    
class BaseTransformer(nn.Module):
    def __init__(self, input_dim, vocab_size, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, max_len=5000, use_cnn=True):
        super().__init__()
        self.d_model = d_model
        self.use_cnn = use_cnn

        if use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(input_dim),
                nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),
                nn.ReLU()
            )

        self.input_fc = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.input_positional = PositionalEncoding(d_model, max_len)

        self.output_embed = nn.Embedding(vocab_size, d_model)
        self.output_positional = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_fc = nn.Linear(d_model, vocab_size)

    def forward(
        self, features, targets,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        memory_mask=None  # (B, T_tgt, T_src) timings selection
    ):
        if self.use_cnn:
            features = features.permute(0, 2, 1)
            features = self.cnn(features)
            features = features.permute(0, 2, 1)

        src = self.input_positional(self.input_dropout(self.input_fc(features)))
        tgt = self.output_positional(self.output_embed(targets) * math.sqrt(self.d_model))

        # Encoder
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)

        # Causal mask
        B, T_tgt, _ = tgt.size()
        causal_mask = torch.triu(torch.ones((T_tgt, T_tgt), device=tgt.device), diagonal=1).bool()
        causal_mask = causal_mask.masked_fill(causal_mask, float('-inf')).masked_fill(~causal_mask, float(0.0))

        # Memory mask
        if memory_mask is not None:
            out = []
            for b in range(B):
                out_b = self.decoder(
                    tgt[b:b+1],
                    memory[b:b+1],
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask[b:b+1] if tgt_key_padding_mask is not None else None,
                    memory_key_padding_mask=src_key_padding_mask[b:b+1] if src_key_padding_mask is not None else None,
                    memory_mask=memory_mask[b]
                )
                out.append(out_b)
            out = torch.cat(out, dim=0)
        else:
            out = self.decoder(
                tgt,
                memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )

        return self.output_fc(out)


    def predict(self, features, max_len=100, start_token=3, end_token=4, src_key_padding_mask=None):
        batch_size = features.size(0)

        if self.use_cnn:
            features = features.permute(0, 2, 1)
            features = self.cnn(features)
            features = features.permute(0, 2, 1)

        src = self.input_positional(self.input_dropout(self.input_fc(features)))
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)

        ys = torch.full((batch_size, 1), start_token, dtype=torch.long, device=features.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=features.device)

        for step in range(max_len - 1):
            tgt = self.output_positional(self.output_embed(ys) * math.sqrt(self.d_model))

            # Causal mask for autoregressive decoding
            tgt_mask = torch.triu(torch.ones((ys.size(1), ys.size(1)), device=features.device), diagonal=1).bool()
            tgt_mask = tgt_mask.masked_fill(tgt_mask == True, float('-inf')).masked_fill(tgt_mask == False, float(0.0))

            out = self.decoder(
                tgt, memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            logits = self.output_fc(out[:, -1:, :])
            next_token = logits.argmax(-1)  # (B, 1)

            next_token[finished.unsqueeze(1)] = end_token
            ys = torch.cat([ys, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == end_token)

            if finished.all():
                break

        output_sequences = []
        for seq in ys.tolist():
            if end_token in seq:
                seq = seq[:seq.index(end_token)]
            output_sequences.append(seq)

        return output_sequences