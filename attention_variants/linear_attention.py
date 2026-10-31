import torch
import torch.nn as nn

class CausalLinearAttention(nn.Module):
    # initialize learnable tensor weights with correct dims
    def __init__(self, d_model, layer_idx=None):
        super().__init__()
        # Zoology gives layer_idx, this attention variant does not need it
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # get batch size (n of sequences), sequence length and d_k (key dimension)
        B, T, d_k = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        # get d_v (value dimension)
        d_v = v.shape[-1]
        batch_outputs = []

        for b in range(B):
            # each sequence starts with its own memory
            S = q.new_zeros(d_k, d_v)
            outputs = []

            for t in range(T):
                # [d_k] outer [d_v] -> [d_k, d_v]
                S = S + torch.outer(k[b, t], v[b, t])
                # [d_k] @ [d_k, d_v] -> [d_v]
                output_t = q[b, t] @ S
                outputs.append(output_t)

            batch_outputs.append(torch.stack(outputs))  # [T, d_v]
        # collect results [B, T, hidden_size]
        o = torch.stack(batch_outputs)
        # transform each token's result
        return self.out_proj(o)
