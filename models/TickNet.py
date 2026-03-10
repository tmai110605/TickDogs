import torch
import torch.nn as nn
import torch.nn.functional as F

# Giả định bạn đã có các hàm cơ bản trong common.py như file gốc
from models.common import conv1x1_block, conv3x3_block

# Tích hợp MAF Attention cải tiến
from models.MAF_Attention import MAF  

# =============================================================================
# 1. TOÁN TỬ TOP (NETTOP) - Trích xuất đặc trưng đa mặt phẳng
# =============================================================================
class TOP_Operator(nn.Module):
    def __init__(self, channels, stride=1):
        super(TOP_Operator, self).__init__()
        self.stride = stride
        
        # Tích chập chiều sâu cho mặt phẳng không gian XY
        self.dw_xy = nn.Conv2d(channels, channels, 3, stride, 1, groups=channels, bias=False)
        self.bn_xy = nn.BatchNorm2d(channels)
        
        # Tích chập chiều sâu cho mặt phẳng XZ và YZ
        self.dw_xz = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.bn_xz = nn.BatchNorm2d(channels)
        
        self.dw_yz = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.bn_yz = nn.BatchNorm2d(channels)
        
        # Dùng AvgPool thay vì MaxPool: 
        # Giúp giữ lại thông tin bối cảnh mượt mà hơn khi giảm chiều (stride=2)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity()

    def forward(self, x):
        # 1. Mặt phẳng XY (Không gian chuẩn: B x C x H x W)
        f_xy = self.bn_xy(self.dw_xy(x))

        # 2. Mặt phẳng XZ (Hoán vị W và C)
        x_xz = x.permute(0, 3, 1, 2) 
        f_xz = self.bn_xz(self.dw_xz(x_xz)).permute(0, 2, 3, 1) 
        f_xz = self.pool(f_xz)

        # 3. Mặt phẳng YZ (Hoán vị H và C)
        x_yz = x.permute(0, 2, 3, 1) 
        f_yz = self.bn_yz(self.dw_yz(x_yz)).permute(0, 3, 1, 2)
        f_yz = self.pool(f_yz)

        # Hợp nhất đặc trưng
        combined = f_xy * torch.sigmoid(f_xz * f_yz)
        
        # SỬ DỤNG RELU TIÊU CHUẨN
        return torch.relu(combined)


# =============================================================================
# 2. KHỐI LAI FR-PDP-TOP
# =============================================================================
class FR_PDP_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(FR_PDP_block, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Lớp Pw1
        self.Pw1 = conv1x1_block(in_channels=in_channels,
                                 out_channels=in_channels,                                
                                 use_bn=False,
                                 activation=None)
        
        # Lớp TOP
        self.TOP = TOP_Operator(channels=in_channels, stride=stride) 
        
        # Lớp Pw2
        self.Pw2 = conv1x1_block(in_channels=in_channels,
                                 out_channels=out_channels,                                             
                                 groups=1)
        
        # Nhánh Identity (Full-Residual)
        self.PwR = conv1x1_block(in_channels=in_channels,
                                 out_channels=out_channels,
                                 stride=stride)
        
        # GỌI MAF ATTENTION THAY VÌ SE
        self.attention = MAF(out_channels, 16) 

    def forward(self, x):
        residual = x
        
        # Nhánh chính
        x = self.Pw1(x)        
        x = self.TOP(x)
        x = self.Pw2(x)
        x = self.attention(x)
        
        # Nhánh Residual kết nối tắt
        if self.stride == 1 and self.in_channels == self.out_channels:
            x = x + residual
        else:            
            residual = self.PwR(residual)
            x = x + residual
            
        return x


# =============================================================================
# 3. KIẾN TRÚC MẠNG STICKNET_TOP_LARGE (15 BLOCKS)
# =============================================================================
class TOP_Operator(nn.Module):
    def __init__(self, channels, stride=1):
        super(TOP_Operator, self).__init__()
        self.stride = stride
        
        # Mặt phẳng XY (Không gian chuẩn 2D: H x W)
        self.dw_xy = nn.Conv2d(channels, channels, 3, stride, 1, groups=channels, bias=False)
        self.bn_xy = nn.BatchNorm2d(channels)
        
        # Mặt phẳng XZ (Dùng Conv1d chạy dọc theo Width, chia sẻ trọng số qua Height)
        self.dw_xz = nn.Conv1d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.bn_xz = nn.BatchNorm2d(channels)
        
        # Mặt phẳng YZ (Dùng Conv1d chạy dọc theo Height, chia sẻ trọng số qua Width)
        self.dw_yz = nn.Conv1d(channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.bn_yz = nn.BatchNorm2d(channels)
        
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride) if stride > 1 else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.size()

        # 1. Mặt phẳng XY [B, C, H, W]
        f_xy = self.bn_xy(self.dw_xy(x))

        # 2. Mặt phẳng XZ 
        # Biến đổi [B, C, H, W] -> [B, H, C, W] -> [B*H, C, W] để chạy Conv1d
        x_xz = x.permute(0, 2, 1, 3).contiguous().view(b * h, c, w)
        f_xz = self.dw_xz(x_xz)
        # Khôi phục lại [B, H, C, W] -> [B, C, H, W]
        f_xz = f_xz.view(b, h, c, w).permute(0, 2, 1, 3).contiguous()
        f_xz = self.pool(self.bn_xz(f_xz))

        # 3. Mặt phẳng YZ
        # Biến đổi [B, C, H, W] -> [B, W, C, H] -> [B*W, C, H] để chạy Conv1d
        x_yz = x.permute(0, 3, 1, 2).contiguous().view(b * w, c, h)
        f_yz = self.dw_yz(x_yz)
        # Khôi phục lại [B, W, C, H] -> [B, C, H, W]
        f_yz = f_yz.view(b, w, c, h).permute(0, 2, 3, 1).contiguous()
        f_yz = self.pool(self.bn_yz(f_yz))

        # Hợp nhất đặc trưng theo công thức NetTOP
        combined = f_xy * torch.sigmoid(f_xz * f_yz)
        return torch.relu(combined)


# =============================================================================
# 4. KIẾN TRÚC MẠNG STICKNET_TOP_SMALL (7 BLOCKS)
# =============================================================================
class STickNet_TOP_Small(nn.Module):
    def __init__(self, num_classes=1000, cifar=False):
        super(STickNet_TOP_Small, self).__init__()
        init_stride = 1 if cifar else 2
        
        # 1. Giai đoạn đầu (Input: 224x224 -> Output: 112x112)
        self.initial = conv3x3_block(in_channels=3, out_channels=32, stride=init_stride)

        # 2. Khối Perceptron bổ sung (Spread-learned spatial features)
        # TỐI ƯU FLOPS: Giữ nguyên số kênh là 32 tại độ phân giải cao (112x112). 
        # Việc này tiết kiệm > 800 Triệu MACs so với việc đẩy lên 256.
        self.extra_perceptron = FR_PDP_block(in_channels=32, out_channels=32, stride=1)

        # 3. Cấu trúc xương sống (Backbone) - Chuẩn Tick-Shape
        self.backbone = nn.Sequential(
            FR_PDP_block(32,  128, stride=2),  # Đi lên Đỉnh 1 (Kích thước giảm còn 56x56)
            FR_PDP_block(128, 64,  stride=1),  # Xuống Đáy 1  (56x56)
            FR_PDP_block(64,  128, stride=1),  # Phục hồi kênh (56x56)
            
            FR_PDP_block(128, 256, stride=2),  # Đi lên Đỉnh 2 (Kích thước giảm còn 28x28)
            FR_PDP_block(256, 128, stride=1),  # Xuống dần    (28x28)
            FR_PDP_block(128, 64,  stride=1),  # Xuống Đáy 2  (28x28)
            
            FR_PDP_block(64,  512, stride=2),  # Tăng tốc cuối (Kích thước giảm còn 14x14)
        )

        # 4. Giai đoạn cuối và phân loại
        self.final_conv = conv1x1_block(in_channels=512, out_channels=1024, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.initial(x)
        x = self.extra_perceptron(x) 
        x = self.backbone(x)         
        x = self.final_conv(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# =============================================================================
# HÀM BUILD MODEL ĐÃ HỖ TRỢ CẢ BẢN LARGE VÀ SMALL
# =============================================================================
def build_TickNet(num_classes=1000, typesize='large_new', cifar=False):
    if typesize in ['large_new', 'large_large_new', 'large']:
        print("Đang khởi tạo STickNet_TOP (LARGE - 15 blocks) với MAF Attention & ReLU...")
        model = STickNet_TOP(num_classes=num_classes, cifar=cifar)
    elif typesize in ['small', 'small_new']:
        print("Đang khởi tạo STickNet_TOP (SMALL - 7 blocks) với MAF Attention & ReLU...")
        model = STickNet_TOP_Small(num_classes=num_classes, cifar=cifar)
    else:
        raise NotImplementedError(f"Phiên bản {typesize} chưa được triển khai với kiến trúc lai mới.")
    
    return model