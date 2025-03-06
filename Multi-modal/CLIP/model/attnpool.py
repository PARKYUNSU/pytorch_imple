import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super(AttentionPool2d, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_size = embed_dim
        self.head_dim = embed_dim // num_heads
        self.all_head_dim = num_heads * self.head_dim

        # Positional embedding for each spatial position
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)

        # Linear projections for query, key, and value
        self.query_dense = nn.Linear(embed_dim, self.all_head_dim)
        self.key_dense = nn.Linear(embed_dim, self.all_head_dim)
        self.value_dense = nn.Linear(embed_dim, self.all_head_dim)
        
        self.attn_dropout = nn.Dropout(0.1)  # Dropout for attention probabilities
        self.output_dense = nn.Linear(embed_dim, output_dim or embed_dim)  # Output projection
        self.proj_dropout = nn.Dropout(0.1)  # Dropout for the final output

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_score(self, x):
        # x: (B, N, all_head_dim) -> (B, num_heads, N, head_dim)
        new_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        B, N, C = x.shape  # B: batch size, N: number of spatial positions, C: channels

        query = self.query_dense(x)
        key = self.key_dense(x)
        value = self.value_dense(x)

        query = self.transpose_for_score(query)
        key = self.transpose_for_score(key)
        value = self.transpose_for_score(value)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        # Apply attention to value
        context = torch.matmul(attention_probs, value).permute(0, 2, 1, 3).contiguous().reshape(B, N, C)

        output = self.output_dense(context)
        output = self.proj_dropout(output)

        return output