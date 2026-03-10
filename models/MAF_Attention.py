import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class MAF_ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(MAF_ChannelGate, self).__init__()        
        
        # Mạng MLP (Bottleneck) giữ nguyên cấu trúc như SE để không làm tăng số lượng tham số
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        
    def forward(self, x):
        b, c, h, w = x.size()
        c_half = c // 2
        
        # BƯỚC 1: Tách kênh (Channel Splitting)
        x_1 = x[:, :c_half, :, :]  # Lấy C/2 kênh đầu
        x_2 = x[:, c_half:, :, :]  # Lấy C/2 kênh sau
        
        # BƯỚC 2: Dual Statistical Pooling
        # Nhánh 1: Global Average Pooling (Lấy giá trị trung bình cường độ)
        avg_pool = F.adaptive_avg_pool2d(x_1, 1) # Kích thước: (B, C/2, 1, 1)
        
        # Nhánh 2: Global Standard Deviation Pooling (Lấy độ phân tán / chi tiết không gian)
        # Reshape x_2 thành (B, C/2, H*W) để tính std trên chiều không gian
        x_2_flat = x_2.view(b, c - c_half, h * w)
        # Tính độ lệch chuẩn (unbiased=False giúp tính toán ổn định hơn)
        std_pool = torch.std(x_2_flat, dim=2, unbiased=False, keepdim=True).view(b, c - c_half, 1, 1)
        
        # BƯỚC 3: Ghép nối (Concatenation)
        # Nối 2 vector lại để khôi phục kích thước (B, C, 1, 1)
        concat_pool = torch.cat([avg_pool, std_pool], dim=1)
        
        # BƯỚC 4 & 5: Đưa qua MLP và tạo trọng số (Reweighting)
        channel_att = self.mlp(concat_pool)
        scale = torch.sigmoid(channel_att).unsqueeze(2).unsqueeze(3).expand_as(x)
        
        return x * scale

class MAF(nn.Module):
    """
    Mixed Attention Fusion (MAF) Block
    Dựa trên ý tưởng cải tiến: Chia đôi kênh cho Avg Pooling và Std Pooling.
    """
    def __init__(self, gate_channels, reduction_ratio=16):
        super(MAF, self).__init__()
        self.ChannelGate = MAF_ChannelGate(gate_channels, reduction_ratio)
        
    def forward(self, x):
        x_out = self.ChannelGate(x)
        return x_out