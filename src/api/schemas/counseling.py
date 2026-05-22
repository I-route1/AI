from pydantic import BaseModel

class ReportGenerationRequest(BaseModel):
    studentId: str
    currentKoreanGrade: float
    studyHours: float
    studentNote: str
    teacherFeedback: str