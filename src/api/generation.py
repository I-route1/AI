"""개념 설명·리포트 생성 설정. 서빙과 평가 스크립트가 같은 값을 쓰도록 한곳에 둔다.

생성 길이 (2026-09-25 측정, 과목별 개념 12개를 한도 800으로 생성):
- Qwen 개념 설명은 자연스럽게 끝나는 데 240~538토큰(중앙값 약 390)이 든다. 이전 한도
  200에서는 12개 전부가 목록 중간에 잘렸다. 이어서 79개 평가를 600으로 돌렸더니 수학 7개가
  마무리 직전에 잘렸다 — LaTeX 수식이 토큰을 많이 쓴다. 그래서 800. 한 번에 약 12~40초
  (약 20토큰/초)가 걸린다 — 이전 약 10초.
- Ollama(llama3.1)는 73~316토큰. 이전 한도 150에서 8개 중 3개가 잘렸다. 400으로 올려도
  1~3초라 부담이 없다.
"""

CONCEPT_GEN = dict(max_new_tokens=800, temperature=0.3, do_sample=True, repetition_penalty=1.2)
# 평가용 결정적 디코딩. 한도와 반복 벌점은 서빙과 같다.
CONCEPT_GEN_GREEDY = dict(max_new_tokens=800, do_sample=False, repetition_penalty=1.2)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:latest"
# num_ctx: 참고 자료가 붙으면 한국어가 2천 토큰 가까이 된다.
OLLAMA_OPTIONS = {"num_predict": 400, "temperature": 0.4, "num_ctx": 4096}
