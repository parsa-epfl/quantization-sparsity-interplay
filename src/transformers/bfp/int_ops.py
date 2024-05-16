import torch

def _quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

def int_quantize_zero_mean(x, bits):
    maxq = 2**(bits-1) - 1
    max_val = torch.max(torch.abs(x))
    return torch.clamp(torch.round(maxq*x/max_val), -maxq, maxq) * (max_val/maxq)

def int_quantize(x, bits, device, identifier):
    maxq = 2**bits-1
    max_val = torch.max(x)
    min_val = torch.min(x)
    scale = (max_val-min_val)/maxq
    zero = (-min_val)/scale
    
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale*(q-zero)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t = torch.randn(5, 6, device=device)

    quant_t = int_quantize(t, 8, device, 'a')
    quant_t_2 = int_quantize_zero_mean(t, 8)
    print(t)
    print(quant_t)
    print(quant_t_2)
