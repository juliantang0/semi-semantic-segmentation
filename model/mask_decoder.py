# -*- coding = utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskDecoder(nn.Module):
    def __init__(self, image_embedding_dim=1000, text_embedding_dim=512, embedding_dim=768, side_len=224,
                 color_channels=3):
        # out_channels = 3 * side_len * side_len
        super(MaskDecoder, self).__init__()
        self.image_embedding_dim = image_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.embedding_dim = embedding_dim
        self.side_len = side_len
        self.color_channels = color_channels
        self.proj1 = nn.Linear(self.image_embedding_dim, self.embedding_dim)
        self.proj2 = nn.Linear(self.text_embedding_dim, self.embedding_dim)
        self.fc1 = nn.Linear(self.image_embedding_dim, self.side_len * self.side_len)
        self.fc2 = nn.Linear(self.side_len * self.side_len, self.side_len * self.side_len * self.color_channels)

    def forward(self, image_embedding, text_embedding):
        attention_scores = torch.matmul(self.proj2(text_embedding), (self.proj1(image_embedding)).T)
        attention_weights = F.softmax(attention_scores, dim=1)
        cross_attended_embedding = torch.matmul(attention_weights, image_embedding)
        out = self.fc1(cross_attended_embedding)
        out = F.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        out = out.reshape(-1, self.color_channels, self.side_len, self.side_len)
        return out
