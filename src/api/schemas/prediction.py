from pydantic import BaseModel

class PredictionRequest(BaseModel):
    studentId: str
    past_score_avg: float
    study_hours_per_day: float

class PredictionResponse(BaseModel):
    expected_score: int
    message: str