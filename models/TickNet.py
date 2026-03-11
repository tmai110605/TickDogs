import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import conv1x1_block, conv3x3_block, Classifier

# =============================================================================
# 1. MIXED ATTENTION FUSION (MAF) - Giữ nguyên từ bản của bạn
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
# 2. CẢI TIẾN: PARTIAL TOP OPERATOR (Trích xuất 3D siêu nhẹ)
# =============================================================================
class Partial_TOP_Operator(nn.Module):
    def __init__(self, channels, stride=1):
        super(Partial_TOP_Operator, self).__init__()
        self.stride = stride
        self.top_channels = channels // 2  # Chỉ dùng TOP cho 1 nửa số kênh
        self.dw_channels = channels - self.top_channels
        
        # Nhánh 1: Depthwise 2D thông thường cho nửa kênh đầu (Rất nhẹ)
        self.dw_standard = nn.Sequential(
            nn.Conv2d(self.dw_channels, self.dw_channels, 3, stride, 1, groups=self.dw_channels, bias=False),
            nn.BatchNorm2d(self.dw_channels)
        )
        
        # Nhánh 2: TOP Operator cho nửa kênh sau (Chi tiết 3D)
        self.dw_xy = nn.Conv2d(self.top_channels, self.top_channels, 3, stride, 1, groups=self.top_channels, bias=False)
        self.bn_xy = nn.BatchNorm2d(self.top_channels)
        
        self.dw_xz = nn.Conv1d(self.top_channels, self.top_channels, 3, 1, 1, groups=self.top_channels, bias=False)
        self.bn_xz = nn.BatchNorm2d(self.top_channels)
        
        self.dw_yz = nn.Conv1d(self.top_channels, self.top_channels, 3, 1, 1, groups=self.top_channels, bias=False)
        self.bn_yz = nn.BatchNorm2d(self.top_channels)
        
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity()

    def forward(self, x):
        # Tách kênh
        x_std, x_top = torch.split(x, [self.dw_channels, self.top_channels], dim=1)
        
        # Xử lý nhánh chuẩn
        out_std = self.dw_standard(x_std)
        
        # Xử lý nhánh TOP
        b, c, h, w = x_top.size()
        f_xy = self.bn_xy(self.dw_xy(x_top))

        x_xz = x_top.permute(0, 2, 1, 3).contiguous().view(b * h, c, w)
        f_xz = self.dw_xz(x_xz).view(b, h, c, w).permute(0, 2, 1, 3).contiguous()
        f_xz = self.pool(self.bn_xz(f_xz))

        x_yz = x_top.permute(0, 3, 1, 2).contiguous().view(b * w, c, h)
        f_yz = self.dw_yz(x_yz).view(b, w, c, h).permute(0, 2, 3, 1).contiguous()
        f_yz = self.pool(self.bn_yz(f_yz))

        out_top = f_xy * torch.sigmoid(f_xz * f_yz)
        
        # Ghép nối lại và kích hoạt
        return torch.relu(torch.cat([out_std, out_top], dim=1))

# =============================================================================
# 3. CẢI TIẾN: SHUFFLE & LITE FR-PDP BLOCK
# =============================================================================
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.size()
        c_per_group = c // self.groups
        x = x.view(b, self.groups, c_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        return x.view(b, -1, h, w)

class Lite_FR_PDP_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups=4):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Đổi thành Grouped Conv 1x1 + Shuffle
        self.Pw1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, groups=groups, bias=False),
            # Không dùng BN sau Pw1 theo thiết kế FR-PDP gốc để giữ nguyên phân phối
        )
        self.shuffle = ChannelShuffle(groups)
        
        # Thay thế TOP bằng Partial TOP siêu nhẹ
        self.TOP_Lite = Partial_TOP_Operator(channels=in_channels, stride=stride)         
        
        # Đổi thành Grouped Conv 1x1
        self.Pw2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Đường tắt Full-Residual
        if self.stride != 1 or self.in_channels != self.out_channels:
            self.PwR = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.PwR = nn.Identity()
            
        # MAF Attention
        self.attention = MAF(out_channels, 16)

    def forward(self, x):
        residual = self.PwR(x)
        
        out = self.Pw1(x)
        out = self.shuffle(out)
        out = self.TOP_Lite(out)        
        out = self.Pw2(out)
        out = self.attention(out)
        
        return F.relu(out + residual, inplace=True)

# =============================================================================
# 4. MẠNG TICKNET (Giữ nguyên cấu trúc nhưng dùng block mới)
# =============================================================================
class TickNet(nn.Module):
    def __init__(self, num_classes, init_conv_channels, init_conv_stride, channels, strides, in_channels=3, use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm

        self.backbone = nn.Sequential()

        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", nn.BatchNorm2d(num_features=in_channels))

        self.backbone.add_module("init_conv", conv3x3_block(in_channels=in_channels, 
                                                            out_channels=init_conv_channels, 
                                                            stride=init_conv_stride))

        # Build Backbone với Lite_FR_PDP_block
        in_c = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = nn.Sequential()
            for unit_id, out_c in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1                
                stage.add_module(f"unit{unit_id + 1}", Lite_FR_PDP_block(in_channels=in_c, out_channels=out_c, stride=stride))
                in_c = out_c
            self.backbone.add_module(f"stage{stage_id + 1}", stage)

        self.final_conv_channels = 1024        
        self.backbone.add_module("final_conv", conv1x1_block(in_channels=in_c, 
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
# 5. HÀM KHỞI TẠO (Đã loại bỏ small_7blocks, tối ưu bản basic và small)
# =============================================================================
def build_TickNet(num_classes, typesize='basic', cifar=False):
    init_conv_channels = 32
    
    if typesize == 'basic':
        # Bản Basic cấu trúc "Dấu tích đơn": 5 blocks, cực nhẹ nhờ nâng cấp Operator
        channels = [[128], [64], [128], [256], [512]] 
        
    elif typesize == 'small':
        # Bản Small tiêu chuẩn: 10 blocks với sức mạnh trích xuất 3D tốt
        channels = [[128], [64, 128], [256, 512, 128], [64, 128, 256], [512]]
        
    elif typesize == 'large':
        # Bản Large (15 blocks)
        channels = [[128], [64, 128], [256, 512, 128, 64, 128, 256], [512, 128, 64, 128, 256], [512]]
    
    else:
        raise ValueError(f"Không hỗ trợ typesize: {typesize}")

    if cifar:
        init_conv_stride = 1
        strides = [1, 1, 2, 2, 2] 
    else:
        init_conv_stride = 2
        if typesize == 'basic':
            strides = [1, 2, 2, 2, 2]
        else:
            strides = [2, 1, 2, 2, 2]

    return TickNet(num_classes=num_classes,
                   init_conv_channels=init_conv_channels,
                   init_conv_stride=init_conv_stride,
                   channels=channels,
                   strides=strides)

