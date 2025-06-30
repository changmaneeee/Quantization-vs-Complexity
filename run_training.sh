#!/bin/bash

# 이 스크립트를 실행하기 전에:
# 1. `prepare_dataset.py`를 실행하여 configs/class_splits.json 파일 생성했는지 확인
# 2. 필요한 라이브러리가 모두 설치되었는지 확인 (conda/pip)

# --- 실험 설정 ---
# 실험할 비트 수 목록 (전체)
BITS_LIST=(1 2 4 8 16 32)
# 실험할 클래스 수 목록 (전체)
CLASSES_LIST=(10 20 50 100)
# 총 학습 에포크 수
EPOCHS=100

echo "==================================================="
echo "  STARTING BATCH TRAINING FOR ALL CONFIGURATIONS   "
echo "==================================================="

# 이중 for 루프로 모든 조합에 대해 실험 실행
for BITS in "${BITS_LIST[@]}"; do
    for CLASSES in "${CLASSES_LIST[@]}"; do
        
        # 현재 실행 중인 실험 정보 출력
        echo ""
        echo "---------------------------------------------------"
        echo ">>> RUNNING: ${BITS}-bit, ${CLASSES}-classes, ${EPOCHS}-epochs"
        echo "---------------------------------------------------"
        
        # train.py 스크립트 실행
        # 각 실험의 표준 출력과 에러를 별도의 로그 파일로 저장
        python train.py \
            --bits "$BITS" \
            --num_classes "$CLASSES" \
            --epochs "$EPOCHS" \
            # --lr 0.001 --batch_size 128 등 다른 인자도 여기서 설정 가능
            > "logs/training/${BITS}bit_${CLASSES}class.log" 2>&1
        
        # 에러 발생 시 스크립트 중단 (선택사항, 하지만 권장)
        if [ $? -ne 0 ]; then
            echo "!!!!!! ERROR: An error occurred during the training of ${BITS}-bit, ${CLASSES}-classes. !!!!!!"
            echo "Please check the log file: logs/training/${BITS}bit_${CLASSES}class.log"
            exit 1
        fi
        
        echo ">>> FINISHED: ${BITS}-bit, ${CLASSES}-classes"
        
    done
done

echo ""
echo "==================================================="
echo "  ALL TRAINING EXPERIMENTS FINISHED SUCCESSFULLY!  "
echo "==================================================="