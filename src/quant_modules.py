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



