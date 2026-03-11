import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        encoded_seq, _ = self.encoder(x)
        decoded_seq, _ = self.decoder(encoded_seq)
        out = self.output_layer(decoded_seq)
        return out