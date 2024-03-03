import torch
import torch.nn as nn
import numpy as np
import math


def get_upsample_filter(size):
    # 创建一个二位双线性核，用于上采样操作，使用双线性滤波器确定新像素的值，用于放大操作
    # 滤波器的影响半径，由 size 决定
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    # 创建坐标网络
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    # 返回一个 filter: 滤波器，用于图像上采样的双线性滤波器
    return torch.from_numpy(filter).float()

# 包含多层卷积和LeakyReLU激活的递归块
class RecursiveBlock(nn.Module):
    def __init__(self, d):
        super(RecursiveBlock, self).__init__()
        # 初始化一个连续的神经网络模块
        self.block = nn.Sequential()
        # 根据参数d，添加 d 个 LeakyReLU 激活层和卷积层
        for i in range(d):
            # 添加LeakyReLU激活层，负斜率为0.2，inplace参数为 True 意味着将直接在输入上进行操作以节省内存
            self.block.add_module("relu_" + str(i), nn.LeakyReLU(0.2, inplace=True))
            # 添加卷积层，输入输出通道数均为64，卷积核大小为3x3，步长为1，填充为1，使用偏置
            self.block.add_module("conv_" + str(i), nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                                                              stride=1, padding=1, bias=True))

    # 定义前向传播函数，输入x通过block模块处理后得到输出
    def forward(self, x):
        output = self.block(x)
        return output

# 使用递归块对输入特征进行多次迭代，实现特征的嵌入
class FeatureEmbedding(nn.Module):
    def __init__(self, r, d):
        super(FeatureEmbedding, self).__init__()

        self.recursive_block = RecursiveBlock(d)
        self.num_recursion = r

    def forward(self, x):
        output = x.clone()

        # The weights are shared within the recursive block!
        for i in range(self.num_recursion):
            output = self.recursive_block(output) + x

        return output

# 通过一个初始卷积层提取特征，然后使用
class LapSrnMS(nn.Module):
    def __init__(self, r, d, scale):
        super(LapSrnMS, self).__init__()

        self.scale = scale
        self.conv_input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, )

        self.transpose = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3,
                                            stride=2, padding=0, bias=True)
        self.relu_features = nn.LeakyReLU(0.2, inplace=True)

        self.scale_img = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4,
                                            stride=2, padding=0, bias=False)

        self.predict = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
        self.features = FeatureEmbedding(r, d)

        i_conv = 0
        i_tconv = 0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if i_conv == 0:
                    m.weight.data = 0.001 * torch.randn(m.weight.shape)
                else:
                    m.weight.data = math.sqrt(2 / (3 * 3 * 64)) * torch.randn(m.weight.shape)
                    # torch.nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')

                i_conv += 1

                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                if i_tconv == 0:
                    m.weight.data = math.sqrt(2 / (3 * 3 * 64)) * torch.randn(m.weight.shape)
                else:
                    c1, c2, h, w = m.weight.data.size()
                    # 初始化 ConvTranspose2d 层的权重为双线性核
                    weight = get_upsample_filter(h)
                    m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)

                i_tconv += 1

                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        features = self.conv_input(x)
        output_images = []
        rescaled_img = x.clone()

        for i in range(int(math.log2(self.scale))):
            # 构建多尺度特征
            features = self.features(features)
            features = self.transpose(self.relu_features(features))
            # 通过转置卷积上采样图像
            features = features[:, :, :-1, :-1]
            rescaled_img = self.scale_img(rescaled_img)
            rescaled_img = rescaled_img[:, :, 1:-1, 1:-1]
            # 通过预测层来生成最终超分辨率图像
            predict = self.predict(features)
            out = torch.add(predict, rescaled_img)

            out = torch.clamp(out, 0.0, 1.0)

            output_images.append(out)

        return output_images


class CharbonnierLoss(nn.Module):
    # L1损失的平滑版本
    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        # print(error)
        loss = torch.sum(error)
        return loss
