# prepare_dataset.py
import json
import numpy as np
import os

def main():
    """
    CIFAR-100 클래스를 무작위로 섞어 서브셋 그룹을 만들고,
    그 결과를 JSON 파일로 저장합니다.
    이 스크립트는 모든 실험 시작 전에 딱 한 번만 실행합니다.
    """
    print("Creating class splits for CIFAR-100...")
    
    config_dir = "configs"
    os.makedirs(config_dir, exist_ok=True)
    
    # 재현성을 위해 난수 시드 고정
    np.random.seed(42)
    
    # 0부터 99까지의 클래스 인덱스 생성 및 셔플
    all_classes = list(range(100))
    np.random.shuffle(all_classes)
    
    # 클래스 수에 따른 그룹 정의
    class_splits = {
        10: all_classes[:10],
        20: all_classes[:20],
        50: all_classes[:50],
        100: all_classes # 전체
    }
    
    # 보기 좋게 정렬해서 저장
    for k, v in class_splits.items():
        class_splits[k] = sorted(v)
    
    file_path = os.path.join(config_dir, "class_splits.json")
    with open(file_path, 'w') as f:
        json.dump(class_splits, f, indent=4)
        
    print(f"Class splits saved to {file_path}")
    print("Sample (10 classes):", class_splits[10])

if __name__ == '__main__':
    main()