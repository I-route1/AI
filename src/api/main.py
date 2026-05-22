from fastapi import FastAPI
from src.api.routers import counseling, predictor

# FastAPI 서버 인스턴스 생성
app = FastAPI(title="i-Route AI Core Server", version="2.0.0")

# 🌟 라우터 조립 (스프링 부트의 Controller 매핑과 동일한 역할)
app.include_router(predictor.router, prefix="/api/ai", tags=["Predictor"])
app.include_router(counseling.router, prefix="/api/ai", tags=["Counseling"])

@app.get("/")
def health_check():
    return {"status": "ok", "message": "i-Route AI Server is running!"}

if __name__ == "__main__":
    import uvicorn
    # 폴더 구조가 바뀌었으므로 실행 경로를 'src.api.main:app'으로 지정합니다.
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8082, reload=True)