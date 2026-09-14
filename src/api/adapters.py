"""베이스 모델과 LoRA 어댑터 경로/프롬프트 설정.

main.py에서 분리해 둔 이유: 평가 스크립트 등에서 이 값들만 필요할 때
main.py를 import하면 8B 모델 로드가 통째로 실행되기 때문이다.
"""

# 베이스 모델 하나(Qwen3-8B)에 과목별 어댑터를 올려 set_adapter()로 갈아끼우며 서빙한다.
# 예전엔 수학=Gemma4, 글쓰기=원격 Llama3.1 어댑터로 서로 다른 베이스라 8B 모델 두 개를
# 동시에 띄워야 했는데(VRAM 부족으로 OOM), 같은 베이스로 재학습해서 하나로 합쳤다.
BASE_MODEL_ID        = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MATH_ADAPTER_PATH    = "train/math_adapter_qwen"
WRITING_ADAPTER_PATH = "train/writing_adapter_qwen_weighted"

MATH_SYSTEM_PROMPT = "당신은 수학 전문 교사입니다. 학생의 수학 문제에 대해 정확한 풀이와 해설을 제공하세요."

# 과목 -> (어댑터 이름, 경로, system 프롬프트)
# system 프롬프트는 train/preprocess_curriculum.py의 SUBJECTS와 글자 단위로 동일해야 한다.
# 학습 때 쓴 문구와 다르면 파인튜닝 효과가 떨어진다.
# 한국사는 AI-Hub 교육과정 데이터에 해당 과목이 없어 어댑터가 없다(Ollama fallback).
SUBJECT_ADAPTERS: dict[str, tuple[str, str, str]] = {
    "국어": ("korean",  "train/korean_adapter_qwen",
             "당신은 국어 전문 교사입니다. 학생의 국어 지문과 질문에 대해 정확하고 이해하기 쉬운 답변을 제공하세요."),
    "영어": ("english", "train/english_adapter_qwen",
             "당신은 영어 전문 교사입니다. 학생의 영어 지문과 질문에 대해 정확하고 이해하기 쉬운 답변을 제공하세요."),
    "과학": ("science", "train/science_adapter_qwen",
             "당신은 과학 전문 교사입니다. 학생의 과학 지문과 질문에 대해 정확하고 이해하기 쉬운 답변을 제공하세요."),
    "사회": ("social",  "train/social_adapter_qwen",
             "당신은 사회 전문 교사입니다. 학생의 사회 지문과 질문에 대해 정확하고 이해하기 쉬운 답변을 제공하세요."),
}
