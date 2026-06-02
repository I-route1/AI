from fastapi import APIRouter, HTTPException
import joblib
import numpy as np
from src.api.schemas.prediction import PredictionRequest, PredictionResponse

router = APIRouter()

# 모델 로드 (경로 주의)
MODEL_PATH = "src/models/score_predictor/score_prediction_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print("[Predictor] 머신러닝 예측 모델 로드 완료!")
except FileNotFoundError:
    print("[Predictor] 모델 파일을 찾을 수 없습니다.")
    model = None

@router.post("/predict", response_model=PredictionResponse)
async def predict_next_score(req: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="AI 예측 모델이 준비되지 않았습니다.")

    try:
        features = np.array([[req.past_score_avg, req.study_hours_per_day]])
        raw = model.predict(features)[0]
        # 모델 출력이 0~1 정규화값이면 100점 척도로 역변환
        predicted_score = int(raw * 100) if raw <= 1.0 else int(raw)
        predicted_score = max(0, min(100, predicted_score))

        diff = predicted_score - req.past_score_avg
        if diff >= 1:
            msg = "📈 현재 학습 패턴이 좋습니다! 성적 향상이 기대됩니다."
        elif diff >= -2:
            msg = "📊 현재 학습량으로 성적을 유지할 수 있습니다. 조금 더 노력하면 더 올릴 수 있어요."
        elif diff >= -4:
            msg = "⚠️ 현재 학습량으로는 소폭 하락이 예상됩니다. 학습 시간을 늘려보세요."
        else:
            msg = "🚨 학습량이 크게 부족합니다. 지금 바로 학습 계획을 점검하세요."

        return PredictionResponse(expected_score=predicted_score, message=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))