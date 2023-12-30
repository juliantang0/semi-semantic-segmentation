# -*- coding = utf-8 -*-
import numpy as np
import torch
from torch import nn

from dataset.VOCdataset import VOCSegmentationDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt

from model.model import Model


def training(model, devices, train_loader, optimizer, loss, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()  # 训练模式 - GCNNet.dropout = True
    for batch_idx, data in enumerate(train_loader):
        image, split_masks, text_embeddings = data['image'], data['split_masks'], data['text_embeddings']
        optimizer.zero_grad()
        image, text_embeddings = image.to(devices[0]), text_embeddings.to(devices[0])
        prediction = model(image, text_embeddings)
        loss = loss(prediction, split_masks)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.y),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


root_dir = 'data/VOCdevkit/VOC2012'

train_dataset = VOCSegmentationDataset(root_dir, split='train')
val_dataset = VOCSegmentationDataset(root_dir, split='val')

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 定义超参数
BATCH_SIZE = 4
NUM_EPOCHS = 10
LOG_INTERVAL = 20
LR = 0.01
device_count = torch.cuda.device_count()
devices = [torch.device(f'cuda:{i}') for i in range(device_count)]

# 定义模型
model = Model()
model = nn.DataParallel(model, device_ids=devices)
model.image_encoder.load_state_dict(torch.load('./checkpoints/mae_pretrain_vit_base.pth'), strict=False)
# 定义损失函数
loss = CrossEntropyLoss()
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 开始训练
for epoch in range(1, NUM_EPOCHS + 1):
    training(model, devices, train_loader, optimizer, loss, epoch)
