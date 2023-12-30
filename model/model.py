# -*- coding = utf-8 -*-
from functools import partial

import torch
import torch.nn as nn
from .image_encoder import VisionTransformer
from .mask_decoder import MaskDecoder


class Model(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768, num_heads=12, side_len=224, num_of_labels=10, color_channels=3):
        super(Model, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.side_len = side_len
        self.num_of_labels = num_of_labels
        self.color_channels = color_channels
        # image encoder
        self.image_encoder = VisionTransformer(
            patch_size=self.patch_size, embed_dim=self.embed_dim, depth=12, num_heads=self.num_heads, mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # mask decoder
        self.mask_decoder = MaskDecoder()

    def forward(self, image, text_embedding):
        batch_size = image.shape[0]
        image_embedding = self.image_encoder(image)
        prediction = torch.zeros(self.num_of_labels, batch_size, self.color_channels, self.side_len, self.side_len)
        text_embedding = text_embedding.permute(1, 0, 2)
        for i in range(self.num_of_labels):
            prediction[i] = self.mask_decoder(image_embedding, text_embedding[i])
        return prediction.permute(1, 0, 2, 3, 4)
