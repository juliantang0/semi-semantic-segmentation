# -*- coding = utf-8 -*-
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel


class CLIP(nn.Module):
    def __init__(self, model_name):
        super(CLIP, self).__init__()
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model = CLIPModel.from_pretrained(model_name)

    def forward(self, text):
        text_features = self.clip_processor(text=text, return_tensors="pt", padding=True)
        text_features = self.clip_model.get_text_features(**text_features)
        return text_features
