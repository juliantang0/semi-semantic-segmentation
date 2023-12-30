# -*- coding = utf-8 -*-
# RGB to label mapping
import numpy as np
import torch

rgb_to_label = {
    (128, 0, 0): "aeroplane",  # airplane
    (0, 128, 0): "bicycle",
    (128, 128, 0): "bird",
    (0, 0, 128): "boat",
    (128, 0, 128): "bottle",
    (0, 128, 128): "bus",
    (128, 128, 128): "car",
    (64, 0, 0): "cat",
    (192, 0, 0): "chair",
    (64, 128, 0): "cow",
    (192, 128, 0): "diningtable",
    (64, 0, 128): "dog",
    (192, 0, 128): "horse",
    (64, 128, 128): "motorbike",
    (192, 128, 128): "person",
    (0, 64, 0): "pottedplant",
    (128, 64, 0): "sheep",
    (0, 192, 0): "sofa",
    (128, 192, 0): "train",
    (0, 64, 128): "tvmonitor",
}


def preprocess(annotated_mask: torch.Tensor):
    annotated_mask = annotated_mask.permute(1, 2, 0)
    ground_truth = {}
    for pixel in annotated_mask.reshape(-1, 3).unique(dim=0):
        pix = (pixel.numpy() * 255).astype(int)
        if tuple(pix) in rgb_to_label.keys():
            label = rgb_to_label[tuple(pix)]
            extracted_tensor = torch.zeros_like(annotated_mask)
            loc = np.all(annotated_mask.numpy() == pixel.numpy(), axis=-1)
            extracted_tensor[loc] = annotated_mask[loc]
            ground_truth[label] = extracted_tensor.permute(2, 0, 1)
    return ground_truth
