# src/model_architectures.py
import torch.nn as nn
import torch.nn.functional as F

# 우리 부품 공장에서 QuantizedConv2d와 QuantizedLinear를 가져옵니다.
from .quant_modules import QuantizedConv2d, QuantizedLinear

# ───────────────────────────────────────────────────────────────
# 1. 양자화된 ResNet 기본 블록 (Quantized Basic Building Block)
# ───────────────────────────────────────────────────────────────

class QuantizedBasicBlock(nn.Module):
    """
    ResNet의 기본 블록(Basic Block)을 양자화 버전으로 정의합니다.
    이 블록은 두 개의 양자화된 컨볼루션 레이어로 구성됩니다.
    """
    expansion = 1 # BasicBlock의 경우 입출력 채널 배수가 1로 동일

    def __init__(self, in_planes, planes, stride=1, bits=8):
        super(QuantizedBasicBlock, self).__init__()
        self.bits = bits

        # 첫 번째 양자화 컨볼루션 레이어 + 배치 정규화
        self.conv1 = QuantizedConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, bits=self.bits
        )
        self.bn1 = nn.BatchNorm2d(planes)

        # 두 번째 양자화 컨볼루션 레이어 + 배치 정규화
        self.conv2 = QuantizedConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, bits=self.bits
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # Shortcut Connection (Identity Mapping)
        # 입력과 출력의 차원(채널 수, 크기)이 다를 경우, shortcut도 맞춰줘야 함
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # 안정성을 위해 shortcut connection은 보통 FP32로 유지합니다.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # BNN이 아닌 이상, 활성화 함수는 일반적으로 ReLU를 사용합니다.
        # (BNN의 경우, 별도의 이진 활성화 함수를 forward에 적용해야 합니다.)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # 잔차(Residual) 연결
        out = F.relu(out)
        return out


# ───────────────────────────────────────────────────────────────
# 2. 일반화된 ResNet 모델 조립기 (The Generic ResNet Builder)
# ───────────────────────────────────────────────────────────────

class ResNet(nn.Module):
    """
    어떤 종류의 블록이든 받아서 ResNet 구조를 조립하는 일반 클래스.
    """
    def __init__(self, block, num_blocks, bits, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.bits = bits # bits 정보는 디버깅이나 로깅에 활용 가능

        # 첫 번째 Conv 레이어: FP32로 유지하여 초기 특징 추출의 정확도 보존
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet의 4개 스테이지(stage) 구성
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # 마지막 Linear(FC) 레이어: FP32로 유지하여 최종 분류의 정확도 보존
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """하나의 스테이지(여러 블록의 연속)를 만드는 헬퍼 함수."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            # block 클래스의 인스턴스를 생성할 때 bits 정보를 전달
            layers.append(block(self.in_planes, planes, s, bits=self.bits))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))) # 32x32 -> 32x32
        
        out = self.layer1(out) # 32x32 -> 32x32
        out = self.layer2(out) # 32x32 -> 16x16
        out = self.layer3(out) # 16x16 -> 8x8
        out = self.layer4(out) # 8x8   -> 4x4
        
        # Adaptive Average Pooling: 입력 크기에 상관없이 출력을 1x1로 만듦
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1) # 1차원 벡터로 펼치기 (Flatten)
        out = self.linear(out)
        return out


# ───────────────────────────────────────────────────────────────
# 3. 모델 생성 팩토리 함수 (The User-Friendly Factory)
# ───────────────────────────────────────────────────────────────

def build_resnet18(bits, num_classes):
    """
    사용자가 원하는 bit-width와 클래스 수에 맞는 ResNet-18 모델을 생성하여 반환합니다.
    이 함수 하나만 외부에서 호출하면 됩니다.
    """
    print(f"Building ResNet-18 with {bits}-bit quantized blocks (num_classes={num_classes}).")
    
    # 모든 비트 수에 대해 동일한 'QuantizedBasicBlock'을 사용.
    # 블록 내부의 QuantizedConv2d가 'bits' 인자에 따라 알아서 다르게 동작함.
    return ResNet(QuantizedBasicBlock, [2, 2, 2, 2], bits=bits, num_classes=num_classes)