#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dcgan_fashionmnist_book.py
--------------------------------
本文件完整复现书中 DCGAN（4.5 节）在 Fashion-MNIST 上的实现（教学/实验版）。
- 代码严格遵循书中结构与超参数（生成器带 BatchNorm，判别器去掉 BatchNorm）。
- 判别器在每次生成器更新前更新 NUM_ITER_D 次（书中实验折衷）。
- 在每个 epoch 结束时用固定噪声生成 100 张样本保存（10x10 网格）。
- 包含书中给出的 FID 计算核心（多元高斯公式），以及如何准备特征的注释。

运行方式：
    python dcgan_fashionmnist_book.py

依赖：
    pip install torch torchvision matplotlib tqdm numpy scipy
    （scipy 只在需要 FID 时才必需）
"""

import os
import time
import math
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

# 若需要计算 FID，使用 scipy.linalg.sqrtm
try:
    from scipy import linalg
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# -----------------------------
# 超参数（默认按书中）
# -----------------------------
IMG_SIZE = 28         # 图像尺寸（宽高），Fashion-MNIST 为 28
BATCH_SIZE = 256      # 批大小（书中使用 256）
DIM_Z = 128           # 噪声向量 z 的维度（书中设 128）
LR_D = 1e-4           # 判别器学习率（书中 1e-4）
LR_G = 1e-4           # 生成器学习率（书中 1e-4）
EPOCHS = 200          # 训练轮数（书中 200）
NUM_ITER_D = 2        # 判别器在每次生成器更新之前更新的次数（书中实践值）
BETA1 = 0.5           # Adam beta1（书中及 DCGAN 常用 0.5）
SAMPLE_NROW = 10      # 采样保存时每行的图片数（10x10 网格）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUT_DIR = './outputs'
CKPT_DIR = './checkpoints'
DATA_ROOT = './data'

# -----------------------------
# 创建目录
# -----------------------------
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------------
# 数据加载与预处理
# -----------------------------
# 书中对 Fashion-MNIST 的处理：ToTensor() + Normalize((0.5,), (0.5,))
# 这会把像素从 [0,1] 映射到 [-1,1]，与 Generator 输出 tanh 对应。
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

# -----------------------------
# 模型定义 —— Generator
# 书中实现：全卷积生成器，包含 5 个卷积块。
# 前 4 个卷积块：每块由一个 4x4 转置卷积、BatchNorm2d、ReLU 组成（特征图尺寸翻倍）
# 最后一个卷积块：3x3 转置卷积 + Tanh，用于输出最终图像（28x28）
# 注意：注释中明确标注每层的特征图尺寸，便于理解维度变化。
# -----------------------------
class Generator(nn.Module):
    def __init__(self, dim_z=DIM_Z, out_channels=1):
        """
        生成器（全卷积）
        :param dim_z: 噪声维度（z）
        :param out_channels: 输出通道数（Fashion-MNIST 为 1）
        """
        super(Generator, self).__init__()
        self.dim_z = dim_z
        self.out_channels = out_channels

        # 下面按照书中代码清单实现（注：注释写明输入/输出特征图维度）
        # 输入 z 的视图: (n, dim_z, 1, 1)
        # 1) ConvTranspose2d(128 -> 512), kernel=4, stride=1, pad=0  -> 输出 (n,512,4,4)
        # 2) ConvTranspose2d(512 -> 256), kernel=4, stride=1, pad=0 -> 输出 (n,256,7,7)  (由公式计算 (4-1)*1 -2*0 +4 = 7)
        # 3) ConvTranspose2d(256 -> 128), kernel=4, stride=2, pad=1 -> 输出 (n,128,14,14)
        # 4) ConvTranspose2d(128 -> 64), kernel=4, stride=2, pad=1  -> 输出 (n,64,28,28)
        # 5) ConvTranspose2d(64 -> out_channels), kernel=3, stride=1, pad=1 -> 输出 (n,1,28,28)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(dim_z, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 输出层：3x3 转置卷积 + Tanh -> 输出范围 [-1, 1]
            nn.ConvTranspose2d(64, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        """
        前向：
        - 输入 input: 形状 (batch_size, dim_z)
        - 先 reshape 为 (batch_size, dim_z, 1, 1)
        - 通过转置卷积序列生成 (batch_size, out_channels, 28, 28)
        """
        x = input.view(-1, self.dim_z, 1, 1)
        return self.conv(x)


# -----------------------------
# 模型定义 —— Discriminator
# 书中实现：全卷积判别器，包含 5 个卷积块。
# 前 4 个卷积块：每块由一个 4x4 卷积 + LeakyReLU 组成（未使用 BatchNorm）
# 最后一个卷积块：1x1 卷积 + Sigmoid 输出（得到 (n,1,1,1)）
# 书中注：未使用 BatchNorm 是因为在 Fashion-MNIST 上，加入 BatchNorm 会让判别器拟合得太快，可能导致训练崩溃。
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        """
        判别器（全卷积）
        :param in_channels: 输入通道数（Fashion-MNIST:1）
        """
        super(Discriminator, self).__init__()
        self.in_channels = in_channels

        # 结构注释：
        # 输入 (n,1,28,28)
        # 1) Conv2d(1 -> 64), k=4, s=2, p=1 -> 输出 (n,64,14,14)
        # 2) Conv2d(64 -> 128), k=4, s=2, p=1 -> 输出 (n,128,7,7)
        # 3) Conv2d(128 -> 256), k=4, s=1, p=0 -> 输出 (n,256,4,4)
        # 4) Conv2d(256 -> 512), k=4, s=1, p=0 -> 输出 (n,512,1,1)
        # 5) Conv2d(512 -> 1), k=1, s=1, p=0 -> 输出 (n,1,1,1) -> Sigmoid -> (n,1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # -> 14x14
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> 7x7
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),  # -> 4x4
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0),  # -> 1x1
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, input):
        out = self.conv(input)  # (n,1,1,1)
        # reshape to (n,1)
        return out.view(-1, 1)


# -----------------------------
# 权重初始化（如书中建议/实验使用）
# 这里我们使用较常见的正态初始化 N(0, 0.02)，这是 DCGAN 推荐的
# -----------------------------
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # 卷积层/转置卷积层权重服从 N(0, 0.02)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        # BatchNorm层的权重接近于 1、偏置为 0
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


# -----------------------------
# FID 计算函数（书中给出的多元高斯公式）
# 注：此函数接收从 Inception 或其他网络得到的特征向量 Xr, Xg（shape: [N, dim]）
# 返回 FID 分数（float）
# 说明：计算过程遵循公式 d^2 = ||mu_r - mu_g||^2 + Tr(C_r + C_g - 2 sqrt(C_r C_g))
#       需要 scipy.linalg.sqrtm 来计算矩阵平方根；若无 scipy，请先 pip install scipy
# -----------------------------
def compute_fid_from_features(Xr, Xg, eps=1e-10):
    """
    计算 FID（Frechet Inception Distance），书中给出的核心公式实现。
    Xr: numpy array, shape (N_r, dim)
    Xg: numpy array, shape (N_g, dim)
    """
    if not _HAS_SCIPY:
        raise RuntimeError("计算 FID 需要 scipy，请安装 scipy：pip install scipy")

    # 1) 计算均值
    mu_r = np.mean(Xr, axis=0)
    mu_g = np.mean(Xg, axis=0)
    # 2) 计算协方差矩阵（按列为变量）
    sigma_r = np.cov(Xr, rowvar=False)
    sigma_g = np.cov(Xg, rowvar=False)

    # 差的平方范数
    diff = mu_r - mu_g
    diff_sq = diff.dot(diff)

    # 计算 sqrtm(sigma_r * sigma_g)
    covmean, _ = linalg.sqrtm(sigma_r.dot(sigma_g), disp=False)
    # 处理数值误差和奇异矩阵情况
    if not np.isfinite(covmean).all():
        print("FID: covmean has non-finite values, adding eps to diagonal of cov estimates")
        offset = np.eye(sigma_r.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_r + offset).dot(sigma_g + offset))
    # 如果 covmean 为复数（数值误差），取其实部
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff_sq + np.trace(sigma_r + sigma_g - 2.0 * covmean)
    return float(fid)


# -----------------------------
# 训练函数（书中代码清单 4-5 的翻译实现）
# 注意：保留书中设计（每次 G 更新前 D 更新 NUM_ITER_D 次）
# -----------------------------
def train(netG, netD, device=DEVICE):
    """
    训练主循环（按书中流程）
    - netG: Generator 实例
    - netD: Discriminator 实例
    - device: 训练设备（cuda 或 cpu）
    """
    netG = netG.to(device)
    netD = netD.to(device)

    # 优化器（书中用 Adam）
    optimizerG = optim.Adam(netG.parameters(), lr=LR_G, betas=(BETA1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=LR_D, betas=(BETA1, 0.999))

    # 判别器使用二元交叉熵损失（书中使用 BCE）
    criterion = nn.BCELoss()

    # 生成固定噪声，用于每个 epoch 生成样本并可视化训练进程
    n_row = SAMPLE_NROW
    z_fixed = torch.randn(n_row * n_row, DIM_Z, dtype=torch.float).to(device)

    start_time = time.time()

    # 日志收集
    G_losses = []
    D_losses = []

    # 训练循环（epoch）
    for epoch in range(EPOCHS):
        netG.train()
        netD.train()

        # 把 train_loader 转为迭代器，书中采用这种 while + next 的方式
        data_iter = iter(train_loader)
        batch_idx = 0
        # while 循环控制：直到使用完一个 epoch 的所有 batch
        while batch_idx < len(train_loader):
            # --------------------------
            # (1) 更新判别器 D: 每更新一次 G，就更新 NUM_ITER_D 次 D（书中设置为 2）
            # --------------------------
            for _ in range(NUM_ITER_D):
                if batch_idx >= len(train_loader):
                    break
                # 读取一个 batch 的真实图像
                batch_train_images, _ = next(data_iter)
                batch_idx += 1

                # batch 大小
                batch_size_current = batch_train_images.shape[0]
                # 转为 float 并移动到设备（同时已在 transform 中归一化到 [-1,1]）
                batch_train_images = batch_train_images.type(torch.float).to(device)

                # 采样噪声并用 G 生成伪造图像
                z = torch.randn(batch_size_current, DIM_Z, dtype=torch.float).to(device)
                gen_imgs = netG(z)  # 生成器输出 (batch, 1, 28, 28)

                # 真实标签为 1，虚假标签为 0（书中没有使用标签平滑）
                real_gt = torch.ones(batch_size_current, 1, device=device)
                fake_gt = torch.zeros(batch_size_current, 1, device=device)

                # D 的训练步骤
                optimizerD.zero_grad()

                # 判别器对真实图像的预测
                prob_real = netD(batch_train_images)  # 形状 (batch,1)
                real_loss = criterion(prob_real, real_gt)

                # 判别器对伪造图像的预测（注意 detach()，不更新 G）
                prob_fake = netD(gen_imgs.detach())
                fake_loss = criterion(prob_fake, fake_gt)

                # 总的 D loss（取平均）
                d_loss = (real_loss + fake_loss) / 2.0
                d_loss.backward()
                optimizerD.step()
                # D 更新结束（若 NUM_ITER_D >1 则 loop 继续，用下一个 batch）

            # --------------------------
            # (2) 更新生成器 G（一轮D更新结束后更新一次G）
            # --------------------------
            # 生成器更新使用新的噪声采样
            optimizerG.zero_grad()
            z = torch.randn(batch_size_current, DIM_Z, dtype=torch.float).to(device)
            gen_imgs = netG(z)
            # 判别器对生成图的输出（这里不 detach，因为我们要更新 G）
            dis_out = netD(gen_imgs)

            # 书中生成器损失（式 (4-22)）对应使用 BCE，将判别器输出与 "real" 标签对齐
            # 等价于非饱和损失（实践稳定），这里用 BCE(real) 即 -log D(G(z))
            g_loss = criterion(dis_out, real_gt)
            g_loss.backward()
            optimizerG.step()

            # 记录损失
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

        # --- 本 epoch 结束，打印日志并用固定噪声生成样本用于可视化 ---
        elapsed = time.time() - start_time
        print("Epoch [{}/{}] | D loss: {:.6f} | G loss: {:.6f} | D prob real {:.4f} | D prob fake {:.4f} | time: {:.1f}s"
              .format(epoch + 1, EPOCHS, d_loss.item(), g_loss.item(),
                      prob_real.mean().item(), prob_fake.mean().item(), elapsed))

        # 保存 checkpoint（按 epoch）
        torch.save({'epoch': epoch + 1,
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimizerG': optimizerG.state_dict(),
                    'optimizerD': optimizerD.state_dict()},
                   os.path.join(CKPT_DIR, f'dcgan_epoch_{epoch+1:03d}.pth'))

        # 用固定噪声 z_fixed 生成 n = n_row*n_row 张图像并保存
        netG.eval()
        with torch.no_grad():
            gen_imgs = netG(z_fixed).detach().cpu()  # 形状 (n_row*n_row, 1, 28, 28)
        # 保存图片网格（normalize=True 将 [-1,1] 映射到 [0,1]）
        save_image(gen_imgs, os.path.join(OUT_DIR, f'epoch_{epoch+1:03d}.png'), nrow=n_row, normalize=True)
        netG.train()

    # 训练结束，保存最终模型与损失曲线
    torch.save(netG.state_dict(), os.path.join(CKPT_DIR, 'netG_final.pth'))
    torch.save(netD.state_dict(), os.path.join(CKPT_DIR, 'netD_final.pth'))

    # 绘制损失曲线并保存（训练过程中的所有批次）
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(D_losses, label='D_loss')
        plt.plot(G_losses, label='G_loss')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend()
        plt.title('DCGAN losses (book implementation)')
        plt.savefig(os.path.join(OUT_DIR, 'loss_curve.png'))
        plt.close()
    except Exception:
        print("绘制损失曲线失败（缺少 matplotlib 或其他错误）。")

    return netG


# -----------------------------
# 采样函数（书中代码清单 4-7）
# 功能：用预训练的 netG 生成指定数量的样本并返回 numpy 数组（n, c, h, w）
# -----------------------------
def sample_netG(netG, nfake, batch_size=100, device=DEVICE):
    """
    - netG: 训练好的生成器
    - nfake: 期望生成样本数量
    - batch_size: 每次生成的批次大小
    返回 numpy 数组 shape (nfake, c, h, w)
    """
    netG = netG.to(device)
    netG.eval()
    output = []
    with torch.no_grad():
        num_got = 0
        while num_got < nfake:
            z = torch.randn(batch_size, DIM_Z, dtype=torch.float).to(device)
            batch_fake_images = netG(z)  # torch tensor (batch_size, 1, 28, 28)
            output.append(batch_fake_images.cpu().numpy())
            num_got += batch_size
    output = np.concatenate(output, axis=0)
    return output[:nfake]  # 取前 nfake 个样本


# -----------------------------
# 主入口：实例化 netG、netD 并训练
# -----------------------------
def main():
    # 打印配置信息（书中参数）
    print("DCGAN (book) config:")
    print(f"IMG_SIZE={IMG_SIZE}, BATCH_SIZE={BATCH_SIZE}, DIM_Z={DIM_Z}, LR_D={LR_D}, LR_G={LR_G}, EPOCHS={EPOCHS}, NUM_ITER_D={NUM_ITER_D}")
    print("Device:", DEVICE)

    # 实例化模型并进行权重初始化（书中建议的初始化可用，这里使用 normal_）
    netG = Generator(dim_z=DIM_Z, out_channels=1)
    netD = Discriminator(in_channels=1)
    netG.apply(weights_init_normal)
    netD.apply(weights_init_normal)

    # 训练并返回训练好的生成器
    trained_G = train(netG, netD, device=DEVICE)

    # 示例：训练完成后生成 10000 张样本并（可选）计算 FID（若你已准备好 Inception 特征）
    # 下面仅示例如何生成样本；真实 FID 计算需要你先用预训练的 InceptionV3 在真实图像和生成图像上分别提取特征（见书中说明）
    print("生成 100 个示例图片并保存为 sample_100.png")
    fake100 = sample_netG(trained_G, 100, batch_size=100, device=DEVICE)
    # fake100 shape (100,1,28,28), 值在 [-1,1]，需要映射回 [0,1] 保存
    fake100_t = torch.from_numpy(fake100)
    save_image(fake100_t, os.path.join(OUT_DIR, 'sample_100.png'), nrow=10, normalize=True)
    print("Done. 输出保存在：", OUT_DIR)


if __name__ == '__main__':
    main()
