import torch
import torch.nn as nn
from torchvision import models

class MultiFrameResNet(nn.Module):
    def __init__(self, num_input_frames=50, num_classes=5):
        """
        基于 ResNet18 修改的视频分类模型
        输入: [Batch, Frames, H, W] -> 修改第一层卷积处理堆叠帧
        输出: [Batch, Num_Classes]
        """
        super(MultiFrameResNet, self).__init__()
        
        # 加载标准 ResNet18
        self.backbone = models.resnet18(pretrained=True)
        
        # --- 修改第一层卷积 (The Paper Trick) ---
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=num_input_frames, # 这里改成 50
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias
        )
        
        # 初始化权重：将RGB权重的平均值复制到50个通道
        with torch.no_grad():
            avg_weight = torch.mean(old_conv.weight, dim=1, keepdim=True)
            new_conv.weight.data = avg_weight.repeat(1, num_input_frames, 1, 1)
            
        self.backbone.conv1 = new_conv
        
        # --- 修改全连接层 (分类头) ---
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: [Batch, Channels, H, W]
        out = self.backbone(x)
        probs = self.softmax(out)
        return probs

# 简单的音频 CNN (复现论文中的轻量级结构)
class SimpleAudioCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleAudioCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return self.softmax(x)