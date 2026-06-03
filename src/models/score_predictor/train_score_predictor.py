"""
성적 예측 모델 훈련 스크립트
입력: past_score_avg (0~100), study_hours_per_day (0~10)
출력: 다음 시험 예상 점수 (0~1 정규화, 서빙 시 ×100)
"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(42)
N = 2000

past_score = np.random.uniform(20, 100, N)
study_hours = np.random.uniform(0.5, 8.0, N)

# 현실적인 점수 변화 공식
# - 기본값: 과거 점수 유지
# - 학습 시간 2시간 미만 → 점수 하락 (최대 -10점)
# - 학습 시간 2~4시간 → 현상 유지 ~ 소폭 상승
# - 학습 시간 4시간 이상 → 점수 상승 (최대 +15점)
study_effect = np.where(
    study_hours < 2.0,
    -10 * (1 - study_hours / 2.0),
    np.where(study_hours < 4.0, (study_hours - 2.0) * 2.5, 5 + (study_hours - 4.0) * 2.0)
)
study_effect = np.clip(study_effect, -10, 15)

# 고득점 구간 천장 효과 (95점 이상은 오르기 어려움)
ceiling_penalty = np.where(past_score > 90, (past_score - 90) * 0.5, 0)

noise = np.random.normal(0, 3, N)

next_score = past_score + study_effect - ceiling_penalty + noise
next_score = np.clip(next_score, 0, 100)

X = np.column_stack([past_score, study_hours])
y = next_score / 100.0  # 0~1 정규화

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test * 100, y_pred * 100)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}점  |  R²: {r2:.4f}")

# 예측 샘플 출력
samples = [
    (50.0, 1.0),
    (72.5, 2.0),
    (80.0, 4.0),
    (60.0, 6.0),
    (95.0, 5.0),
]
print("\n[예측 샘플]")
for avg, hours in samples:
    pred = int(model.predict([[avg, hours]])[0] * 100)
    print(f"  평균 {avg}점 / 하루 {hours}시간 → 예측: {pred}점")

output_path = Path(__file__).parent / "score_prediction_model.pkl"
joblib.dump(model, output_path)
print(f"\n모델 저장 완료: {output_path}")
