import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):#, only_diff=True):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.init_hidden()
        combined = torch.cat((input, hidden), 0)
        hidden = torch.tanh(self.i2h(combined))
        output = self.i2o(combined)

        # if(self.only_diff):
        #     output = output + input[-len(output):]
        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.hidden_size)