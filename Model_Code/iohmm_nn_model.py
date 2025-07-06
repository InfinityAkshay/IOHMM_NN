import torch
import torch.nn as nn
import torch.nn.functional as F

def convolve(A, B):
    C = torch.tensordot(A, B, dims=[[-1], [0]])
    return C

def forward_step(input_t, model, prev_state_prob = None):
    if(prev_state_prob == None):
        current_state_prob = F.softmax(model.W_pi @ input_t + model.b_pi, dim=0)
    else:
        transition_matrix = F.softmax(convolve(model.W_z, input_t) + model.b_z, dim = 0)
        current_state_prob = transition_matrix @ prev_state_prob

    X = convolve( model.W_x, input_t ) + model.b_x

    output_t = X @ current_state_prob
    # if( model.only_diff ):
    #     output_t = output_t + input_t[-len(output_t):]
    
    return output_t, current_state_prob


class IOHMM_NN(nn.Module):

    def __init__(self, input_dim, output_dim, num_states = 5, add_bias = True):#, only_diff = True):
        super().__init__()

        self.W_pi = nn.Parameter(torch.randn(num_states, input_dim ))
        self.W_z  = nn.Parameter(torch.randn(num_states, num_states, input_dim))
        self.W_x  = nn.Parameter(torch.randn(output_dim, num_states, input_dim))

        if add_bias:
            self.b_pi = nn.Parameter(torch.zeros(num_states))
            self.b_z  = nn.Parameter(torch.zeros(num_states, num_states))
            self.b_x  = nn.Parameter(torch.zeros(output_dim, num_states))
        else:
            self.b_pi = torch.zeros(num_states)
            self.b_z  = torch.zeros(num_states, num_states)
            self.b_x  = torch.zeros(output_dim, num_states)


    def forward(self, input_t, prev_state_prob = None):
        return forward_step(input_t, self, prev_state_prob)