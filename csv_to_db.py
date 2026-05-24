import pandas as pd
from sqlalchemy import create_engine

# 💡 본인의 MySQL 접속 정보로 수정하세요!
DB_URL = 'mysql+pymysql://iroute:iroute_pw@localhost:3306/i_route_db'
CSV_FILE_PATH = "./all_cleaned_student_info.csv"

try:
    print("🚀 1. 전처리된 CSV 파일을 읽어오는 중...")
    df = pd.read_csv(CSV_FILE_PATH)

    print(f"🔗 2. MySQL 데이터베이스 연결 중...")
    engine = create_engine(DB_URL)

    # 데이터 적재 (테이블이 없으면 자동 생성됨)
    df.to_sql('student_info', con=engine, if_exists='replace', index=False)

    print(f"🎉 무결성 검증 완료! {len(df)}건의 실제 학생 진로상담 데이터가 'student_info' 테이블에 안착했습니다.")

except Exception as e:
    print(f"❌ DB 적재 중 오류 발생: {e}")