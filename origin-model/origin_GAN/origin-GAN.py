#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vanilla GAN (Goodfellow et al., 2014) 原始实现（含中文注释）
-----------------------------------------------------
主要内容：
- 使用MNIST数据集，复现原始GAN的min-max对抗训练框架
- 保留论文中的两种生成器损失形式：minimax（原始）和non-saturating（改进）
- 默认使用SGD优化器，以贴近论文；可选择Adam获得更稳定结果
*- python origin-GAN.py --epochs 30 --batch_size 128
- python origin-GAN.py --optimizer adam --gen_loss non_saturating --epochs 50(更加稳定训练-Adam + non-saturating)
- python origin-GAN.py --optimizer adam --gen_loss non_saturating --lr_adam 0.00005 --epochs 30(借鉴DCGAN)
- python origin-GAN.py \
    --optimizer adam \
    --gen_loss non_saturating \
    --lr_adam 0.00005 \
    --epochs 50

"""

import os
import argparse
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

# ============================================================
# 工具函数
# ============================================================

def make_dirs(path):
    """若目录不存在则创建"""
    Path(path).mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    """固定随机种子，保证结果可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# 模型定义：Generator 和 Discriminator
# ============================================================

class Generator(nn.Module):
    """
    生成器 G：将随机噪声 z 映射为图像。
    结构：多层感知机（MLP）
    激活函数：ReLU（隐藏层），Tanh（输出层，范围[-1,1]）
    """
    def __init__(self, nz=100, hidden=256, out_dim=28*28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nz, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden*2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden*2, hidden*4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden*4, out_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """
    判别器 D：输入图像 x，输出“真假概率” D(x)。
    结构：多层感知机（MLP）
    激活函数：LeakyReLU（隐藏层），Sigmoid（输出层）
    """
    def __init__(self, in_dim=28*28, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden*4, hidden*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden*2, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)

# ============================================================
# 权重初始化函数
# ============================================================

def weights_init(m):
    """Xavier初始化（论文虽未指定，但该方式更稳定）"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# ============================================================
# 生成样本保存函数
# ============================================================

def save_sample_grid(G, fixed_z, out_path, device, nrow=8):
    """
    从固定噪声生成图像并保存为网格。
    作用：可视化生成器随训练过程的变化。
    """
    G.eval()
    with torch.no_grad():
        imgs = G(fixed_z.to(device)).cpu()
    G.train()
    imgs = imgs.view(-1, 1, 28, 28)
    grid = utils.make_grid(imgs, nrow=nrow, normalize=True, value_range=(-1,1))
    utils.save_image(grid, out_path)

# ============================================================
# 训练主循环
# ============================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"使用设备：{device}")
    set_seed(args.seed)
    make_dirs(args.out_dir)
    make_dirs(args.ckpt_dir)

    # -------------------------------
    # 1️⃣ 加载MNIST数据集
    # -------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 将像素缩放到[-1,1]
    ])
    dataset = datasets.MNIST(root=args.data_dir, train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    # -------------------------------
    # 2️⃣ 初始化模型
    # -------------------------------
    G = Generator(nz=args.nz, hidden=args.hidden).to(device)
    D = Discriminator(in_dim=28*28, hidden=args.hidden).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    # -------------------------------
    # 3️⃣ 定义损失函数与优化器
    # -------------------------------
    bce = nn.BCELoss()  # 二分类交叉熵

    if args.optimizer == 'sgd':
        optimizerD = optim.SGD(D.parameters(), lr=args.lr, momentum=args.momentum)
        optimizerG = optim.SGD(G.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizerD = optim.Adam(D.parameters(), lr=args.lr_adam, betas=(0.5, 0.999))
        optimizerG = optim.Adam(G.parameters(), lr=args.lr_adam, betas=(0.5, 0.999))

    # 固定噪声，用于观察训练进度
    fixed_z = torch.randn(64, args.nz)
    real_label = 0.9
    fake_label = 0.0

    G_losses = []
    D_losses = []
    step = 0

    print("开始训练 ...")
    for epoch in range(2, args.epochs + 1):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, (imgs, _) in pbar:
            step += 1
            imgs = imgs.view(imgs.size(0), -1).to(device)

            # =====================================================
            # (1) 更新判别器 D
            # 目标：最大化 log D(x) + log(1 - D(G(z)))
            # =====================================================
            # 真实样本
            labels_real = torch.full((imgs.size(0), 1), real_label, device=device)
            out_real = D(imgs)
            loss_real = bce(out_real, labels_real)

            # 伪造样本
            z = torch.randn(imgs.size(0), args.nz, device=device)
            fake_imgs = G(z)
            labels_fake = torch.full((imgs.size(0), 1), fake_label, device=device)
            out_fake = D(fake_imgs.detach())  # detach防止梯度传回G
            loss_fake = bce(out_fake, labels_fake)

            lossD = loss_real + loss_fake
            lossD.backward()
            optimizerD.step()

            # =====================================================
            # (2) 更新生成器 G
            # 两种损失形式：
            #   - minimax:    最小化 E[log(1 - D(G(z)))]  ← 原论文形式
            #   - non-sat:    最小化 -E[log D(G(z))]      ← 稳定替代形式
            # =====================================================
            G.zero_grad()
            # 希望生成样本被判为“真”
            labels_real = torch.full((imgs.size(0), 1), real_label, device=device)
            out_fake_for_G = D(fake_imgs)

            if args.gen_loss == 'minimax':
                labels_fake_for_minimax = torch.full((imgs.size(0), 1), fake_label, device=device)
                lossG = bce(out_fake_for_G, labels_fake_for_minimax)
            else:
                lossG = bce(out_fake_for_G, labels_real)

            lossG.backward()
            optimizerG.step()

            # 记录损失
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())

            # 每隔若干步保存生成样本
            if step % args.sample_interval == 0:
                save_sample_grid(G, fixed_z, os.path.join(args.out_dir, f'samples_step{step:06d}.png'), device)

            # 打印进度
            if step % 100 == 0:
                pbar.set_description(f"Epoch {epoch} Step {step} | D_loss={lossD.item():.4f} | G_loss={lossG.item():.4f}")

    # =====================================================
    # 训练完成，保存模型与可视化结果
    # =====================================================
    torch.save(G.state_dict(), os.path.join(args.ckpt_dir, 'G_final.pth'))
    torch.save(D.state_dict(), os.path.join(args.ckpt_dir, 'D_final.pth'))

    plt.figure(figsize=(8,5))
    plt.plot(D_losses, label='D_loss')
    plt.plot(G_losses, label='G_loss')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.savefig(os.path.join(args.out_dir, 'loss_curve.png'))
    plt.close()

    save_sample_grid(G, fixed_z, os.path.join(args.out_dir, 'final_samples.png'), device)
    print("训练完成 ✅ 结果已保存至:", args.out_dir)

# ============================================================
# 参数设置
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Vanilla GAN (Goodfellow, 2014) PyTorch实现")
    parser.add_argument('--data_dir', type=str, default='./data', help='MNIST数据集目录')
    parser.add_argument('--out_dir', type=str, default='./outputs', help='生成样本保存目录')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='模型保存目录')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=128, help='批大小')
    parser.add_argument('--nz', type=int, default=100, help='噪声维度')
    parser.add_argument('--hidden', type=int, default=256, help='隐藏层宽度')
    parser.add_argument('--optimizer', choices=['sgd', 'adam'], default='sgd', help='优化器类型')
    parser.add_argument('--lr', type=float, default=0.01, help='SGD学习率')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD动量参数')
    parser.add_argument('--lr_adam', type=float, default=2e-4, help='Adam学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no_cuda', action='store_true', help='禁用CUDA')
    parser.add_argument('--gen_loss', choices=['minimax', 'non_saturating'], default='minimax',
                        help='生成器损失类型（minimax或non_saturating）')
    parser.add_argument('--sample_interval', type=int, default=500, help='样本保存间隔')
    return parser.parse_args()

# ============================================================
# 主入口
# ============================================================

if __name__ == '__main__':
    args = parse_args()
    train(args)
