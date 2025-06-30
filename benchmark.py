# benchmark.py
import argparse
import os
import time
import torch
import pandas as pd
from codecarbon import EmissionsTracker # 에너지 측정을 위해

from src.data_utils import get_dataloader
from src.model_architectures import build_resnet18

def measure_inference_time(model, device, iterations=100):
    """모델의 평균 추론 시간을 측정합니다."""
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32).to(device)

    # 워밍업
    for _ in range(10):
        _ = model(dummy_input)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    avg_time_ms = ((end_time - start_time) / iterations) * 1000
    return avg_time_ms

def measure_accuracy(model, loader, device):
    """모델의 정확도를 측정합니다."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 모델 로드
    model = build_resnet18(bits=args.bits, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.to(DEVICE)
    
    # 2. 데이터 로더 준비 (정확도 측정용)
    test_loader = get_dataloader('cifar100', args.num_classes, 256, train=False)

    # 3. 지표 측정
    print("Measuring accuracy...")
    accuracy = measure_accuracy(model, test_loader, DEVICE)
    
    print("Measuring inference time...")
    inference_time = measure_inference_time(model, DEVICE)
    
    print("Measuring energy consumption...")
    # CodeCarbon으로 추론 1회에 대한 에너지 측정
    tracker = EmissionsTracker(project_name=f"bench_{args.bits}b_{args.num_classes}c", output_dir=args.log_dir, log_level='error')
    tracker.start()
    with torch.no_grad():
        model(torch.randn(1, 3, 32, 32).to(DEVICE))
    emissions_data = tracker.stop()
    # 에너지 소비량 (kWh)을 줄(Joule) 단위로 변환 (1 kWh = 3.6e6 J)
    energy_joules = emissions_data.energy_consumed * 3.6e6
    
    # 4. 결과 정리
    result = {
        'bits': args.bits,
        'num_classes': args.num_classes,
        'accuracy': accuracy,
        'inference_time_ms': inference_time,
        'energy_joules': energy_joules,
        'model_size_mb': os.path.getsize(args.model_path) / (1024 * 1024)
    }
    
    print("Benchmark Result:", result)
    
    # 5. 결과를 CSV 파일에 추가
    df = pd.DataFrame([result])
    if not os.path.exists(args.output_csv):
        df.to_csv(args.output_csv, index=False)
    else:
        df.to_csv(args.output_csv, mode='a', header=False, index=False)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark Trained Models')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--bits', type=int, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--output_csv', type=str, default='./results/benchmark_results.csv')
    parser.add_argument('--log_dir', type=str, default='./logs/benchmarking')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    main(args)