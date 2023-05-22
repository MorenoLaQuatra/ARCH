from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPoolingHead(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionPoolingHead, self).__init__()
        
        self.embed_dim = embed_dim
        self.attention_layer = nn.Linear(embed_dim, 1)
        
    def forward(self, input_tensor):
        # input_tensor: (batch_size, seq_length, embed_dim)
        
        # Compute attention scores
        attention_scores = self.attention_layer(input_tensor).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=-1)
        # Apply attention weights to input tensor
        weighted_input = torch.bmm(input_tensor.transpose(1, 2), attention_weights.unsqueeze(-1)).squeeze(-1)
        
        # Return pooled output
        return weighted_input

class AttentionPoolingClassifier(nn.Module):
    """
    Attention pooling classifier.

    Args:
        embed_dim: The embedding dimension.
        attention_heads: The number of attention heads.
        dropout: The dropout rate.
        num_classes: The number of classes.
    """

    def __init__(self, embed_dim, num_classes):
        super().__init__()

        self.attention_pooling_head = AttentionPoolingHead(embed_dim)
        self.classification_layer = nn.Linear(embed_dim, num_classes)

    def forward(self, input_tensor):
        # input_tensor: (batch_size, seq_length, embed_dim)

        # Apply attention pooling
        pooled_output = self.attention_pooling_head(input_tensor)

        # Apply classification layer
        logits = self.classification_layer(pooled_output)

        return logits