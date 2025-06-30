# train.py
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# 우리 모듈들 임포트
from src.data_utils import get_dataloader
from src.model_architectures import build_resnet18
from src.quant_modules import clip_weights

# ───────────────────────────────────────────────────────────────
# 학습 및 평가 함수
# ───────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, bits_to_clip):
    """한 에포크 동안 모델을 학습시키는 함수."""
    model.train() # 모델을 학습 모드로 설정
    running_loss, correct, total = 0.0, 0, 0
    
    progress_bar = tqdm(loader, desc="Training", leave=False, ncols=100)

    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 이전 배치의 그래디언트 초기화
        optimizer.zero_grad()

        # AMP(Automatic Mixed Precision) 컨텍스트
        # use_amp가 True(16-bit)일 때만 활성화됨
        with autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # GradScaler를 사용하여 역전파
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # TNN-style (<= 4bit) 모델의 경우 가중치 클리핑
        if bits_to_clip <= 4:
            clip_weights(model)
        
        # 통계 기록
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.2f}%")
    
    return running_loss / total, 100 * correct / total

def evaluate_model(model, loader, criterion, device, use_amp):
    """한 에포크 종료 후 모델을 평가하는 함수."""
    model.eval() # 모델을 평가 모드로 설정
    running_loss, correct, total = 0.0, 0, 0
    
    # 평가 시에는 그래디언트 계산이 필요 없음
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating", leave=False, ncols=100)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / total, 100 * correct / total

# ───────────────────────────────────────────────────────────────
# 메인 실행 함수
# ───────────────────────────────────────────────────────────────

def main(args):
    """스크립트의 메인 실행 로직."""
    
    # 1. 환경 설정
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed) # 모든 GPU에 대해 시드 고정
    
    # 결과 저장용 폴더 생성
    log_dir = os.path.join(args.log_dir, f"{args.bits}bit_{args.num_classes}class")
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"--- Starting Experiment ---")
    print(f"  Bit-width: {args.bits}")
    print(f"  Classes: {args.num_classes}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {DEVICE}")
    print("-------------------------")

    # 2. 데이터 로더 준비
    print("Loading dataset...")
    train_loader = get_dataloader('cifar100', args.num_classes, args.batch_size, train=True, num_workers=args.num_workers)
    test_loader = get_dataloader('cifar100', args.num_classes, args.batch_size, train=False, num_workers=args.num_workers)
    print(f"Train loader size: {len(train_loader.dataset)}, Test loader size: {len(test_loader.dataset)}")

    # 3. 모델 생성
    model = build_resnet18(bits=args.bits, num_classes=args.num_classes).to(DEVICE)

    # 4. 학습 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    USE_AMP = (args.bits == 16)
    scaler = GradScaler(enabled=USE_AMP)

    # 5. 학습 및 평가 루프
    print("\nStarting training loop...")
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_test_acc = 0.0

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, DEVICE, USE_AMP, args.bits)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, DEVICE, USE_AMP)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}")

        # 최고 성능 모델 저장
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_path = os.path.join(args.model_save_dir, f"resnet18_{args.bits}bit_{args.num_classes}class_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"🎉 New best model saved to {best_model_path} (Acc: {best_test_acc:.2f}%)")

        scheduler.step()

    total_time = time.time() - start_time
    print(f"\n--- Training Finished in {total_time/60:.2f} minutes ---")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")

    # 6. 최종 결과 저장
    # 마지막 에포크 모델 저장
    final_model_path = os.path.join(args.model_save_dir, f"resnet18_{args.bits}bit_{args.num_classes}class_final.pth")
    torch.save(model.state_dict(), final_model_path)
    
    # 학습 곡선 그래프 저장
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.title(f'Loss Curve ({args.bits}-bit, {args.num_classes}-class)')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.title(f'Accuracy Curve ({args.bits}-bit, {args.num_classes}-class)')
    
    plot_filename = os.path.join(log_dir, 'training_curves.png')
    plt.savefig(plot_filename)
    plt.close() # 스크립트 실행이 멈추지 않도록 닫아줌
    print(f"Training curves saved to {plot_filename}")

# ───────────────────────────────────────────────────────────────
# 스크립트 실행 지점 (Entry Point)
# ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # argparse를 사용하여 커맨드 라인에서 인자를 받음
    parser = argparse.ArgumentParser(description='Unified Training Script for Quantization Analysis')
    
    # 필수 인자
    parser.add_argument('--bits', type=int, required=True, choices=[1, 2, 4, 8, 16, 32], 
                        help='Bit-width for quantization.')
    parser.add_argument('--num_classes', type=int, required=True, choices=[10, 20, 50, 100], 
                        help='Number of classes for the task.')
    
    # 선택적 인자 (기본값 설정)
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader.')
    parser.add_argument('--model_save_dir', type=str, default='./models', help='Directory to save trained models.')
    parser.add_argument('--log_dir', type=str, default='./logs/training', help='Directory to save logs and plots.')
    
    args = parser.parse_args()
    main(args)