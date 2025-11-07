import torch
import torch.nn as nn

class PianoModel(nn.Module):

    d_model = 512  # Dimensionality of the model
    nhead = 8      # Number of attention heads
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048

    def __init__(self):
        super(PianoModel, self).__init__()
        self.transformer_model = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward
        )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt):
        x = self.sigmoid(self.transformer_model(src, tgt))
        x = torch.flatten(x, start_dim=1)
        output = self.softmax(x)
        return output

