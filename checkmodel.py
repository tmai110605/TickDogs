
from torchsummary import summary
from models.TickNet import *
import torch
from ptflops import get_model_complexity_info

# Định nghĩa hàm để tính toán và in ra thông số
# Hàm này nhận mô hình và kích thước input làm tham số
def print_model_stats(model, input_size=(3, 224, 224), print_per_layer=False):
    """
    Tính toán và in ra MACs, FLOPs, và Parameters của mô hình.
    """
    
    print('-' * 70)
    print(f'*** PHÂN TÍCH ĐỘ PHỨC TẠP MÔ HÌNH (Input: {input_size}) ***')
    
    try:
        # Tính toán MACs và Parameters
        # Chú ý: Đã bỏ with torch.cuda.device(0): để tránh xung đột device
        macs, params = get_model_complexity_info(model, input_size, as_strings=True,
                                                 print_per_layer_stat=print_per_layer, 
                                                 verbose=False, 
                                                 flops_units='GMac')
        
        print('{:<35} {:<15}'.format('Computational complexity (MACs): ', macs))
        
        # Tính FLOPs (Dựa trên logic cũ của bạn)
        macs1 = macs.split()
        if len(macs1) > 1:
            # Logic: (MACs value / 2) + unit
            strmacs1 = str(float(macs1[0]) / 2) + ' ' + macs1[1]
            print('{:<35} {:<15}'.format('Floating-point operations (FLOPs): ', strmacs1))
        else:
             print('{:<35} {:<15}'.format('Floating-point operations (FLOPs): ', 'N/A'))
        
        print('{:<35} {:<15}'.format('Number of parameters (ptflops): ', params))
        
    except Exception as e:
        print(f"WARNING: Không thể tính toán MACs/FLOPs. Lỗi: {e}")
        
    # In tham số tính bằng tay
    param_count = sum([p.data.nelement() for p in model.parameters()])
    print('{:<35} {:<15}'.format('Number of model parameters (Manual): ', f'{param_count:,}'))
        
    print('-' * 70)


