# -*- coding = utf-8 -*-
import os

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from .text_encoder import CLIP
from .utils import preprocess


class VOCSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', num_of_labels=10, side_len=224):
        self.root_dir = root_dir
        self.num_of_labels = num_of_labels
        self.side_len = side_len
        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(root_dir, 'SegmentationClass')

        # Read the split file to determine train or validation set
        split_file_path = os.path.join(root_dir, 'ImageSets/Segmentation', f'{split}.txt')
        with open(split_file_path, 'r') as f:
            self.image_list = [line.strip() for line in f.readlines()]

        # Define transformations for image and mask resizing
        self.transform = transforms.Compose([
            transforms.Resize((self.side_len, self.side_len)),
            transforms.ToTensor(),
        ])

        self.text_encoder = CLIP(model_name="./checkpoints/clip-vit-base-patch16")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Load image
        img_name = os.path.join(self.image_dir, f'{self.image_list[idx]}.jpg')
        image = Image.open(img_name).convert('RGB')

        # Load mask
        mask_name = os.path.join(self.mask_dir, f'{self.image_list[idx]}.png')
        mask = Image.open(mask_name).convert('RGB')

        # preprocess mask
        mask_dict = preprocess(self.transform(mask))
        text_embeddings = torch.zeros(self.num_of_labels, 512)
        split_masks = torch.zeros(self.num_of_labels, 3, self.side_len, self.side_len)
        for i, (key, value) in enumerate(mask_dict.items()):
            text_embeddings[i] = self.text_encoder(key)
            split_masks[i] = value

        return {'image': self.transform(image), 'split_masks': split_masks, 'text_embeddings': text_embeddings}
