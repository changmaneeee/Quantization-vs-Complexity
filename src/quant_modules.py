#src/quant_modules.py
#this file contains the quantization modules used in the model(ResNet18)


import torch
import torch.nn as nn
import torch.nn.functional as F

#=========================================================================
#통합 양자화 모듈(Integrated Quantization Module)
#=========================================================================

class QuantizeSTE(torch.autograd.Function):
    """
    하나의 일반화된 STE(Straight-Through Estimator) 함수.
    'bits' 인자에 따라 적절한 양자화 방식을 적용합니다.
    """
    @staticmethod
    def forward(ctx, tensor, bits):
        if bits >= 16:
            return tensor  # 16비트 이상은 양자화하지 않음
        
        #BNN Using
        if bits ==1:
            alpha = tensor.abs().max()
            return alpha * tensor.sign()
        #TNN Using
        if bits == 2:
            delta = 0.7*tensor.abs().mean()
            output = torch.zeros_like(tensor)
            output[tensor > delta] = 1.0
            output[tensor < -delta] = -1.0
            #추후 여기 scaling factor를 적용 가능(bnn alpha처럼)
            return output
        # 4-bit & 8-bit 양자화
        q_min = 0
        q_max = 2.**bits -1
        
        # 1. scale과 zero_point 계산
        t_min, t_max = tensor.min(), tensor.max()
        scale = (t_max - t_min) / (q_max - q_min)
        if scale == 0.0:
            return tensor  # scale이 0이면 양자화하지 않음
        zero_point = q_min - t_min / scale
        zero_point = torch.clamp(zero_point, q_min, q_max).round()

        # 2. 양자화 및 역양자화(Quantize & De-quantize)
        q_tensor = torch.round(tensor / scale + zero_point)
        q_tensor.clamp_(q_min, q_max)  # 양자화된 값은 q_min과 q_max 사이로 제한
        deq_tensor = (q_tensor - zero_point) * scale
        return deq_tensor
    

    @staticmethod
    def backward(ctx, grad_output):
        #STE: 기울기를 그대로 통과
        #BITS 인자는 학습되지 않음으로 기울기 없음
        return grad_output, None

# ───────────────────────────────────────────────────────────────
# 2. 일반화된 양자화 레이어 (The Generic Quantized Building Blocks)
# ───────────────────────────────────────────────────────────────

class QuantizedConv2d(nn.Conv2d):
    """
    일반화된 양자화 컨볼루션 레이어.
    'bits' 인자에 따라 동작 방식이 결정됩니다.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, bits=8):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bits = bits

        # 매우 낮은 비트(<=4)는 학습 안정성을 위해 별도의 실수형 가중치(weight_fp)를 둠
        if self.bits <= 4:
            self.weight_fp = nn.Parameter(self.weight.data.clone())

    def forward(self, x):
        # 16, 32-bit는 일반 컨볼루션과 동일하게 동작
        if self.bits >= 16:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)
        
        # 학습 전략에 따라 양자화할 가중치를 선택
        if self.bits <= 4:
            # 1, 2, 4-bit는 안정적인 weight_fp를 양자화
            weight_to_quantize = self.weight_fp
        else: # 8-bit
            # 8-bit는 QAT처럼 원본 weight를 직접 양자화 시뮬레이션
            weight_to_quantize = self.weight

        quantized_weight = QuantizeSTE.apply(weight_to_quantize, self.bits)
        return F.conv2d(x, quantized_weight, self.bias, self.stride, self.padding)
    
    def reset_parameters(self):
        super().reset_parameters()
        # weight_fp를 사용하는 모델의 경우, 이 값도 함께 초기화
        if self.bits <= 4 and hasattr(self, 'weight_fp'):
            self.weight_fp.data = self.weight.data.clone()

class QuantizedLinear(nn.Linear):
    """일반화된 양자화 선형 레이어."""
    def __init__(self, in_features, out_features, bias=True, bits=8):
        super().__init__(in_features, out_features, bias=bias)
        self.bits = bits
        if self.bits <= 4:
            self.weight_fp = nn.Parameter(self.weight.data.clone())

    def forward(self, x):
        if self.bits >= 16:
            return F.linear(x, self.weight, self.bias)
        
        if self.bits <= 4:
            weight_to_quantize = self.weight_fp
        else:
            weight_to_quantize = self.weight
            
        quantized_weight = QuantizeSTE.apply(weight_to_quantize, self.bits)
        return F.linear(x, quantized_weight, self.bias)
    
    def reset_parameters(self):
        super().reset_parameters()
        if self.bits <= 4 and hasattr(self, 'weight_fp'):
            self.weight_fp.data = self.weight.data.clone()

# ───────────────────────────────────────────────────────────────
# 3. 보조 함수 (Helper Function)
# ───────────────────────────────────────────────────────────────

def clip_weights(model, min_val=-1.0, max_val=1.0):
    """
    weight_fp를 사용하는 저비트 모델의 가중치를 클리핑하여 학습 안정화.
    """
    for module in model.modules():
        # weight_fp를 가진 레이어에만 적용
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)) and module.bits <= 4:
            if hasattr(module, 'weight_fp'):
                module.weight_fp.data.clamp_(min_val, max_val)

