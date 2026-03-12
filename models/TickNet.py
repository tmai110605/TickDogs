import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F

# Đảm bảo bạn đã có các hàm cơ bản trong common.py
from .common import conv1x1_block, conv3x3_block, conv3x3_dw_block, Classifier, conv1x1_group_block
from .SE_Attention import SE

# =============================================================================
# 1. TOÁN TỬ TOP (TOP_Operator) - Phiên bản linh hoạt, không phụ thuộc in_size
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
        
        # Dùng AvgPool để giảm chiều không gian trên các trục nếu stride > 1
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.size()

        # 1. Trích xuất mặt phẳng XY
        f_xy = self.bn_xy(self.dw_xy(x))

        # 2. Trích xuất mặt phẳng XZ
        x_xz = x.permute(0, 2, 1, 3).contiguous().view(b * h, c, w)
        f_xz = self.dw_xz(x_xz)
        f_xz = f_xz.view(b, h, c, w).permute(0, 2, 1, 3).contiguous()
        f_xz = self.pool(self.bn_xz(f_xz))

        # 3. Trích xuất mặt phẳng YZ
        x_yz = x.permute(0, 3, 1, 2).contiguous().view(b * w, c, h)
        f_yz = self.dw_yz(x_yz)
        f_yz = f_yz.view(b, w, c, h).permute(0, 2, 3, 1).contiguous()
        f_yz = self.pool(self.bn_yz(f_yz))

        # 4. Hợp nhất 3 mặt phẳng (Fusion)
        combined = f_xy * torch.sigmoid(f_xz * f_yz)
        return torch.relu(combined)

# =============================================================================
# 2. KHỐI CƠ BẢN NETTOP (Sử dụng TOP_Operator)
# =============================================================================
class TOP_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups=2):
        super(TOP_Block, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Pw1: Tích chập nhóm (Grouped Conv 1x1)
        self.pw1 = conv1x1_group_block(in_channels=in_channels, 
                                       out_channels=in_channels, 
                                       use_bn=False, 
                                       groups=groups, 
                                       activation=None)
        
        # Toán tử TOP 3D siêu việt
        self.top_operator = TOP_Operator(channels=in_channels, stride=stride)
        
        # Pw2: Tích chập nhóm mở rộng kênh
        self.pw2 = conv1x1_group_block(in_channels=in_channels, 
                                       out_channels=out_channels, 
                                       groups=groups)
        
        # SE Attention
        self.SE = SE(out_channels, 16)        

        # Đường Residual tắt
        if stride == 2 or self.in_channels != self.out_channels:
            self.pointRes = conv1x1_group_block(in_channels=in_channels, 
                                                out_channels=out_channels, 
                                                stride=stride, 
                                                groups=groups)
        else:
            self.pointRes = nn.Identity()

    def forward(self, x):
        residual = self.pointRes(x)
        
        out = self.pw1(x)
        out = self.top_operator(out)
        out = self.pw2(out)
        out = self.SE(out)
        
        return out + residual

# =============================================================================
# 3. KIẾN TRÚC XƯƠNG SỐNG NETTOP
# =============================================================================
class NetTOP(nn.Module):
    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 channels,
                 strides,
                 in_channels=3,
                 use_data_batchnorm=True,
                 groups=2):
        super(NetTOP, self).__init__()
        self.use_data_batchnorm = use_data_batchnorm

        self.backbone = nn.Sequential()

        # Data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", nn.BatchNorm2d(num_features=in_channels))

        # Init conv
        self.backbone.add_module("init_conv", conv3x3_block(in_channels=in_channels, 
                                                            out_channels=init_conv_channels, 
                                                            stride=init_conv_stride))

        # Xây dựng các Stages với TOP_Block mới
        in_c = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1
                # Lưu ý: Không cần truyền in_size vào TOP_Block nữa
                stage.add_module(f"unit{unit_id + 1}", TOP_Block(in_channels=in_c, 
                                                                 out_channels=unit_channels, 
                                                                 stride=stride,
                                                                 groups=groups))
                in_c = unit_channels
            self.backbone.add_module(f"stage{stage_id + 1}", stage)                
        
        self.final_conv_channels = 1024
        self.backbone.add_module("final_conv", conv1x1_block(in_channels=in_c, 
                                                             out_channels=self.final_conv_channels, 
                                                             activation="relu"))
        self.backbone.add_module("global_pool", nn.AdaptiveAvgPool2d(output_size=1))

        # Classifier
        self.classifier = Classifier(in_channels=self.final_conv_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        # Khởi tạo trọng số cho Backbone
        for name, module in self.backbone.named_modules():            
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)                        
            elif isinstance(module, nn.Linear):                
                module.weight.data.normal_(0, 0.01)
                module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):                
                module.weight.data.fill_(1)
                module.bias.data.zero_()            
        # Khởi tạo Classifier
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

# =============================================================================
# 4. HÀM KHỞI TẠO MÔ HÌNH (BUILDER)
# =============================================================================
def build_NetTOP(num_classes, cifar=False, groups=2):
    init_conv_channels = 32
    # Cấu trúc kênh backbone nguyên bản của NetTOP
    channels = [[64], [64, 128, 128], [256, 256, 256], [512, 512, 512], [512]]

    if cifar:
        init_conv_stride = 1
        strides = [1, 1, 2, 2, 2]
    else:
        init_conv_stride = 2
        strides = [1, 2, 2, 2, 2]

    # Loại bỏ in_size khỏi hàm khởi tạo
    return NetTOP(num_classes=num_classes,
                  init_conv_channels=init_conv_channels,
                  init_conv_stride=init_conv_stride,
                  channels=channels,
                  strides=strides,
                  groups=groups)

