"""
AI-Hub 학습태도 데이터로 집중도(집중/집중하지않음/졸음) 분류기 학습.

라즈베리파이5에는 이 스크립트가 아니라 학습 결과물(model.joblib)만 올라간다.
Pi5 쪽에서는 dlib 68점 랜드마크 검출기로 실시간 face_points를 뽑아
extract_features()와 동일한 방식으로 정규화한 뒤 이 모델에 입력한다.
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))
from rpi.dataset import CATEGORY_NAMES, load_dataset

TRAIN_DIR = Path(r"D:\학습태도 및 성향 관찰 데이터\3.개방데이터\1.데이터\Training\02.라벨링데이터")
VAL_DIR   = Path(r"D:\학습태도 및 성향 관찰 데이터\3.개방데이터\1.데이터\Validation\02.라벨링데이터")

MODEL_OUT = Path(__file__).parent / "attention_model.joblib"

SAMPLE_PER_ZIP_TRAIN = None  # 전체 데이터 사용 (파인튜닝: 서브샘플링 없이 최종 모델 학습)
SAMPLE_PER_ZIP_VAL   = None


def main():
    t0 = time.time()
    print("[1/4] Training 라벨 로드 중...")
    X_train, y_train = load_dataset(TRAIN_DIR, sample_per_zip=SAMPLE_PER_ZIP_TRAIN)
    print(f"  총 {len(X_train)}건, 소요 {time.time()-t0:.1f}s")

    print("[2/4] Validation 라벨 로드 중...")
    t1 = time.time()
    X_val, y_val = load_dataset(VAL_DIR, sample_per_zip=SAMPLE_PER_ZIP_VAL)
    print(f"  총 {len(X_val)}건, 소요 {time.time()-t1:.1f}s")

    print(f"\n클래스 분포 (train): {np.bincount(y_train)}")
    print(f"클래스 분포 (val)  : {np.bincount(y_val)}")

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=150, max_depth=20, class_weight="balanced", n_jobs=-1, random_state=42
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True, random_state=42
        ),
    }

    print("\n[3/4] 모델 학습 및 검증...")
    best_name, best_model, best_acc = None, None, -1.0
    for name, clf in candidates.items():
        t2 = time.time()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        acc = (pred == y_val).mean()
        print(f"\n--- {name} (학습 {time.time()-t2:.1f}s) ---")
        print(f"검증 정확도: {acc:.4f}")
        print(classification_report(y_val, pred, target_names=CATEGORY_NAMES, digits=3))
        print("혼동행렬:\n", confusion_matrix(y_val, pred))
        if acc > best_acc:
            best_name, best_model, best_acc = name, clf, acc

    print(f"\n[4/4] 최고 모델: {best_name} (val acc={best_acc:.4f}) → {MODEL_OUT}")
    joblib.dump({"model": best_model, "model_name": best_name, "val_acc": best_acc,
                 "category_names": CATEGORY_NAMES}, MODEL_OUT)
    print(f"총 소요 시간: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
