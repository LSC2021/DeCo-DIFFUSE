import torch

import torch.nn as nn

def _init_lstm_state(num_layers, bidirectional, batch_size, hidden_size, ref_tensor):
    num_directions = 2 if bidirectional else 1
    shape = (num_directions * num_layers, batch_size, hidden_size)
    h_0 = torch.zeros(shape, device=ref_tensor.device, dtype=ref_tensor.dtype)
    c_0 = torch.zeros(shape, device=ref_tensor.device, dtype=ref_tensor.dtype)
    return h_0, c_0

class LSTMB_cum(nn.Module):
    """Bidirectional LSTM with an MLP head and cumulative-sum output.

    Each timestep is processed by a bidirectional LSTM, projected through a
    small two-layer MLP, and then accumulated with torch.cumsum so the model
    predicts the running posture trajectory instead of independent increments.
    """

    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super().__init__()
        self.output_size = output_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.2, bidirectional=True)
        self.fc_1 =  nn.Linear(2*hidden_size, 128) # fully connected
        self.fc_2 = nn.Linear(128, output_size) # fully connected last layer
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0, c_0 = _init_lstm_state(
            num_layers=self.num_layers,
            bidirectional=True,
            batch_size=x.size(0),
            hidden_size=self.hidden_size,
            ref_tensor=x,
        )
        output, _ = self.lstm(x, (h_0, c_0))
        out = self.fc_1(output)
        out = self.relu(out) # relu
        out = self.fc_2(out) # final output
        out = torch.cumsum(out, dim=1)
        return out
