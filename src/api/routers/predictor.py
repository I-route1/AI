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
        predicted_score = int(model.predict(features)[0])

        if predicted_score < req.past_score_avg:
            msg = "⚠️ 학습량 부족으로 성적 하락이 예상됩니다. 일일 학습 시간을 늘려보세요!"
        else:
            msg = "📈 현재 학습 패턴이 아주 좋습니다! 목표 달성이 코앞입니다."

        return PredictionResponse(expected_score=predicted_score, message=msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))