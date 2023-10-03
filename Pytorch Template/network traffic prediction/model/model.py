import torch.nn as nn
import torch.nn.functional as F
import torch
from base import BaseModel


class LSTM(nn.Module):
  def __init__(self, n_features, n_hidden, seq_len, n_layers, dropout=0.2): 
      super(LSTM, self).__init__()
      self.dtype = torch.float32
      self.n_hidden = n_hidden
      self.seq_len = seq_len
      self.n_layers = n_layers
      self.lstm = nn.LSTM(
          input_size=n_features,
          hidden_size=n_hidden,
          num_layers=n_layers,
          dropout = dropout
      )
      self.linear = nn.Linear(in_features=n_hidden, out_features=1)
  def reset_hidden_state(self, *args):
      self.hidden = (
          torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
          torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
      )
  def forward(self, sequences):
      batch_size, seq_len = sequences.size()
      self.reset_hidden_state(batch_size)

      sequences = sequences.to(dtype=self.dtype)
      self.hidden = (self.hidden[0].to(dtype=self.dtype), self.hidden[1].to(dtype=self.dtype))

      lstm_out, self.hidden = self.lstm(
          sequences.view(len(sequences), self.seq_len, -1),
          self.hidden
      )
      last_time_step = lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
      y_pred = self.linear(last_time_step)
      return y_pred
