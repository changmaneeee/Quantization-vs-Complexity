# src/data_utils.py
import json
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

# 데이터셋별 평균과 표준편차 (미리 계산된 값)
STATS = {
    'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}

# 데이터셋 경로
DATA_PATH = "./data"
CONFIG_PATH = "./configs/class_splits.json"

def get_dataloader(dataset_name, num_classes, batch_size, train=True, num_workers=4):
    """
    지정된 데이터셋과 클래스 수에 맞는 데이터로더를 반환하는 메인 함수.
    """
    if dataset_name.lower() != 'cifar100':
        raise ValueError("Currently, only 'cifar100' is supported for subset creation.")

    # 1. 데이터 변환(Transform) 정의
    mean, std = STATS[dataset_name.lower()]
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
    # 2. 원본 데이터셋 로드
    full_dataset = datasets.CIFAR100(root=DATA_PATH, train=train, download=True, transform=transform)

    # 3. 클래스 서브셋에 해당하는 인덱스 필터링
    with open(CONFIG_PATH, 'r') as f:
        class_splits = json.load(f)
    
    # JSON 파일의 키는 문자열이므로 변환 필요
    selected_classes = class_splits[str(num_classes)]

    # 클래스 라벨을 0부터 연속적으로 매핑하기 위한 딕셔너리
    # 예: {1: 0, 7: 1, 12: 2, ...}
    class_map = {original_class: new_class for new_class, original_class in enumerate(selected_classes)}
    
    indices = []
    # full_dataset.targets는 라벨 리스트
    for i, target in enumerate(full_dataset.targets):
        if target in selected_classes:
            indices.append(i)
            # 원본 라벨을 새로운 라벨로 교체
            full_dataset.targets[i] = class_map[target]
            
    subset_dataset = Subset(full_dataset, indices)
    
    # 4. 데이터로더 생성
    data_loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=train, # 학습 데이터만 셔플
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return data_loader