import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDeltaNet(nn.Module):
    # initialize learnable tensor weights with correct dims
    def __init__(self, d_model, layer_idx=None):
        super().__init__()
        # Zoology gives layer_idx, this attention variant does not need it
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.beta_proj = nn.Linear(d_model, 1, bias=False)
        self.alpha_proj = nn.Linear(d_model, 1)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # get batch size (n of sequences), sequence length and d_k (key dimension)
        B, T, d_k = x.shape

        q = F.normalize(F.silu(self.W_q(x)), dim=-1)
        k = F.normalize(F.silu(self.W_k(x)), dim=-1)
        v = self.W_v(x)
        # get d_v  (value dimension)
        d_v = v.shape[-1]
        # assign value of beta and alpha for each token (alpha has simplified initialization)
        beta = torch.sigmoid(self.beta_proj(x))  # [B, T, 1]
        alpha = torch.sigmoid(self.alpha_proj(x))  # [B, T, 1]

        batch_outputs = []

        for b in range(B):
            # each sequence starts with its own memory
            S = q.new_zeros(d_k, d_v)
            outputs = []

            for t in range(T):
                # forget part of the previous memory
                S = alpha[b, t, 0] * S
                # read the memory prediction for current key
                v_hat = k[b, t] @ S      # [d_k] @ [d_k, d_v] -> [d_v]
                # calculate prediction error
                error = v[b, t] - v_hat
                # write a correction
                S = S + beta[b, t, 0] * torch.outer(k[b, t], error)
                
                output_t = q[b, t] @ S    # [d_k] @ [d_k, d_v] -> [d_v]
                outputs.append(output_t)

            batch_outputs.append(torch.stack(outputs))  # [T, d_v]
        # collect results [B, T, hidden_size]
        o = torch.stack(batch_outputs)
        # transform each token's result
        return self.out_proj(o)
