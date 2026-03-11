import torch
import torch.nn as nn
import torch.nn.functional as F

# Giả định bạn đã có các hàm cơ bản trong common.py
from .common import conv1x1_block, conv3x3_block, Classifier

# =============================================================================
# 1. MIXED ATTENTION FUSION (MAF)
# =============================================================================
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class MAF_ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(MAF_ChannelGate, self).__init__()        
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        
    def forward(self, x):
        b, c, h, w = x.size()
        c_half = c // 2
        
        x_1 = x[:, :c_half, :, :]
        x_2 = x[:, c_half:, :, :]
        
        avg_pool = F.adaptive_avg_pool2d(x_1, 1)
        
        x_2_flat = x_2.contiguous().view(b, c - c_half, h * w)
        std_pool = torch.std(x_2_flat, dim=2, unbiased=False, keepdim=True).view(b, c - c_half, 1, 1)
        
        concat_pool = torch.cat([avg_pool, std_pool], dim=1)
        channel_att = self.mlp(concat_pool)
        scale = torch.sigmoid(channel_att).unsqueeze(2).unsqueeze(3).expand_as(x)
        
        return x * scale

class MAF(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(MAF, self).__init__()
        self.ChannelGate = MAF_ChannelGate(gate_channels, reduction_ratio)
        
    def forward(self, x):
        return self.ChannelGate(x)


# =============================================================================
# 2. TOÁN TỬ NETTOP (Trích xuất đa mặt phẳng 3D)
# =============================================================================
class TOP_Operator(nn.Module):
    def __init__(self, channels, stride=1):
        super(TOP_Operator, self).__init__()
        self.stride = stride
        
        # Mặt phẳng XY (Conv2d)
        self.dw_xy = nn.Conv2d(channels, channels, 3, stride, 1, groups=channels, bias=False)
        self.bn_xy = nn.BatchNorm2d(channels)
        
        # Mặt phẳng XZ và YZ (Conv1d)
        self.dw_xz = nn.Conv1d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.bn_xz = nn.BatchNorm2d(channels)
        
        self.dw_yz = nn.Conv1d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.bn_yz = nn.BatchNorm2d(channels)
        
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.size()

        # 1. XY
        f_xy = self.bn_xy(self.dw_xy(x))

        # 2. XZ
        x_xz = x.permute(0, 2, 1, 3).contiguous().view(b * h, c, w)
        f_xz = self.dw_xz(x_xz)
        f_xz = f_xz.view(b, h, c, w).permute(0, 2, 1, 3).contiguous()
        f_xz = self.pool(self.bn_xz(f_xz))

        # 3. YZ
        x_yz = x.permute(0, 3, 1, 2).contiguous().view(b * w, c, h)
        f_yz = self.dw_yz(x_yz)
        f_yz = f_yz.view(b, w, c, h).permute(0, 2, 3, 1).contiguous()
        f_yz = self.pool(self.bn_yz(f_yz))

        # Kết hợp và kích hoạt
        combined = f_xy * torch.sigmoid(f_xz * f_yz)
        return torch.relu(combined)


# =============================================================================
# 3. KHỐI LÕI CẢI TIẾN (Tích hợp TOP và MAF)
# =============================================================================
class FR_PDP_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.Pw1 = conv1x1_block(in_channels=in_channels, out_channels=in_channels, use_bn=False, activation=None)
        
        # SỬ DỤNG NETTOP THAY CHO DEPTHWISE GỐC
        self.TOP = TOP_Operator(channels=in_channels, stride=stride)         
        
        self.Pw2 = conv1x1_block(in_channels=in_channels, out_channels=out_channels, groups=1)
        self.PwR = conv1x1_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        
        # SỬ DỤNG MAF THAY CHO SE GỐC
        self.attention = MAF(out_channels, 16)

    def forward(self, x):
        residual = x
        x = self.Pw1(x)        
        x = self.TOP(x)        
        x = self.Pw2(x)
        x = self.attention(x)
        
        if self.stride == 1 and self.in_channels == self.out_channels:
            x = x + residual
        else:            
            residual = self.PwR(residual)
            x = x + residual
        return x


# =============================================================================
# 4. MẠNG TICKNET ĐỘNG (Xây dựng thông qua vòng lặp)
# =============================================================================
class TickNet(nn.Module):
    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 channels,
                 strides,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size

        self.backbone = nn.Sequential()

        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", nn.BatchNorm2d(num_features=in_channels))

        self.backbone.add_module("init_conv", conv3x3_block(in_channels=in_channels, 
                                                            out_channels=init_conv_channels, 
                                                            stride=init_conv_stride))

        # XÂY DỰNG BACKBONE TỰ ĐỘNG DỰA VÀO MẢNG "CHANNELS" VÀ "STRIDES"
        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1                
                stage.add_module(f"unit{unit_id + 1}", FR_PDP_block(in_channels=in_channels, 
                                                                    out_channels=unit_channels, 
                                                                    stride=stride))
                in_channels = unit_channels
            self.backbone.add_module(f"stage{stage_id + 1}", stage)

        self.final_conv_channels = 1024        
        self.backbone.add_module("final_conv", conv1x1_block(in_channels=in_channels, 
                                                             out_channels=self.final_conv_channels, 
                                                             activation="relu"))
        self.backbone.add_module("global_pool", nn.AdaptiveAvgPool2d(output_size=1))
        
        self.classifier = Classifier(in_channels=self.final_conv_channels, num_classes=num_classes)
        self.init_params()

    def init_params(self):
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


# =============================================================================
# 5. HÀM KHỞI TẠO MẠNG (BUILDER)
# =============================================================================
def build_TickNet(num_classes, typesize='small', cifar=False):
    init_conv_channels = 32
    
    # Định nghĩa cấu hình mảng channels linh hoạt
    if typesize == 'basic':
        channels = [[128], [64], [128], [256], [512]] # 5 blocks
        
    elif typesize == 'small':
        # Bản small gốc (10 blocks)
        channels = [[128], [64, 128], [256, 512, 128], [64, 128, 256], [512]]
        
    elif typesize == 'small_7blocks':
        # Kiến trúc 7 blocks mà bạn muốn dùng "extra perceptron"
        # 1 block đầu giữ kênh 32 (Stem) + Xương sống dấu tích 6 block
        channels = [[32], [128, 64, 128], [256, 128, 64], [512]] 
        
    elif typesize == 'large':
        # Bản large gốc (15 blocks)
        channels = [[128], [64, 128], [256, 512, 128, 64, 128, 256], [512, 128, 64, 128, 256], [512]]
    
    else:
        raise ValueError(f"Không hỗ trợ typesize: {typesize}")

    # Xử lý Stride và In_size
    if cifar:
        in_size = (32, 32)
        init_conv_stride = 1
        
        # Khớp số lượng stride array với số lượng stage thực tế
        if typesize == 'small_7blocks':
            strides = [1, 2, 2, 2] # 4 stages
        else:
            strides = [1, 1, 2, 2, 2] # 5 stages
    else:
        in_size = (224, 224)
        init_conv_stride = 2
        
        if typesize == 'basic':
            strides = [1, 2, 2, 2, 2]
        elif typesize == 'small_7blocks':
            strides = [1, 2, 2, 2] # 4 stages
        else:
            strides = [2, 1, 2, 2, 2]

    return TickNet(num_classes=num_classes,
                   init_conv_channels=init_conv_channels,
                   init_conv_stride=init_conv_stride,
                   channels=channels,
                   strides=strides,
                   in_size=in_size)
