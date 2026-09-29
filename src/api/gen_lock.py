import threading

# main.py와 writing.py가 base_model/tokenizer 하나를 공유한다. 동시 요청이
# model.generate()·model.set_adapter()를 겹쳐 부르면 서로의 어댑터 설정을 덮고,
# tokenizer()를 겹쳐 부르면 HF Rust 토크나이저가 "Already borrowed"로 실패한다.
GEN_LOCK = threading.Lock()
TOK_LOCK = threading.Lock()
