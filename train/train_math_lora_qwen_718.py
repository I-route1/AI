"""
수학 LoRA 파인튜닝 - 71718(수학 과목 문제생성 데이터) 버전

train_math_lora_qwen_cot.py와 **하이퍼파라미터가 완전히 같고 데이터만 다르다.**
(배치 16, 2 epoch, LR 2e-4, MAX_LENGTH 768, LoRA r=16. 학습 스텝은 약 1,250 대 1,216.)
데이터만 바꿔야 결과가 달라졌을 때 그 원인을 데이터라고 말할 수 있다.

데이터 (train/preprocess_math_718.py로 만든 math718_train.jsonl, 10,000건)
  - 중학교 1학년~고등학교 공통수학만. 풀이가 계산 과정이다(등호 3개 이상 73.8%,
    풀이 중앙값 207자). 기존 71859 CoT 데이터는 39.5%, 93자였다.
  - 선택형의 보기 번호를 본문의 보기로 값으로 바꿔 '정답: <값>'으로 학습한다.
  - 71859 평가 문항과 유사도 0.9 이상인 문항은 뺐다(평가 오염 방지).

왜 재보는가: 71859 미세조정은 추론을 끈 베이스 위에 아무것도 더하지 못했다
(CoT 어댑터와 베이스가 14 대 14). 71718은 풀이가 훨씬 풍부하고 보기가 본문에 있어
조건이 다르다. 결과가 어떻게 나와도 서빙 어댑터는 교체하지 않는다 - 서빙에는 수학 풀이의
소비처가 없다.

실행: ./venv_blackwell/Scripts/python.exe train/train_math_lora_qwen_718.py
출력: train/math_adapter_qwen_718/
"""

import os
import json
import math
import time
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.amp import autocast

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── 설정 ──────────────────────────────────────────────
BASE_MODEL_ID  = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
TRAIN_FILE     = "train/math718_train.jsonl"
EVAL_FILE      = "train/math718_eval.jsonl"
OUTPUT_DIR     = "train/math_adapter_qwen_718"
MAX_LENGTH     = 768         # preprocess_math_718.py의 --max-length와 일치해야 한다
BATCH_SIZE     = 1
GRAD_ACCUM     = 16
NUM_EPOCHS     = 2
LR             = 2e-4
WARMUP_RATIO   = 0.05
LOG_STEPS      = 1
EVAL_STEPS     = 100
SAVE_STEPS     = 200


class JsonlChatDataset(Dataset):
    def __init__(self, filepath: str, tokenizer, max_length: int):
        self.samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"  토크나이징 중... ({len(lines)}개)")
        for i, line in enumerate(lines):
            obj = json.loads(line)
            messages = obj["messages"]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            prompt_text = tokenizer.apply_chat_template(
                messages[:2], tokenize=False, add_generation_prompt=True
            )
            full_enc = tokenizer(
                full_text, truncation=True, max_length=max_length,
                padding=False, return_tensors=None,
            )
            prompt_ids = tokenizer(
                prompt_text, truncation=True, max_length=max_length,
                padding=False, return_tensors=None,
            )["input_ids"]
            prompt_len = min(len(prompt_ids), len(full_enc["input_ids"]))
            self.samples.append({
                "input_ids": full_enc["input_ids"],
                "attention_mask": full_enc["attention_mask"],
                "prompt_len": prompt_len,
            })
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(lines)} 완료")
        print(f"  토크나이징 완료: {len(self.samples)}개")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_cosine_lr(step: int, total_steps: int, warmup_steps: int, lr: float) -> float:
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr * (1 + math.cos(math.pi * progress)) / 2


def load_train_state(output_dir: str):
    state_path = os.path.join(output_dir, "train_state.json")
    if not os.path.exists(state_path):
        return None, 0, float("inf")
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    return output_dir, state["global_step"], state["best_val_loss"]


def save_train_state(output_dir: str, global_step: int, best_val_loss: float):
    state_path = os.path.join(output_dir, "train_state.json")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump({"global_step": global_step, "best_val_loss": best_val_loss}, f)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    print("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("모델 로드 중 (4bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    for param in model.parameters():
        param.requires_grad = False
    model.enable_input_require_grads()

    resume_dir, resume_step, resume_best_loss = load_train_state(OUTPUT_DIR)
    if resume_dir:
        print(f"체크포인트에서 재개: {resume_dir} (step {resume_step}, best_val_loss {resume_best_loss:.4f})")
        model = PeftModel.from_pretrained(model, resume_dir, is_trainable=True)
    else:
        model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    print("학습 데이터 로드 중...")
    train_dataset = JsonlChatDataset(TRAIN_FILE, tokenizer, MAX_LENGTH)
    print("평가 데이터 로드 중...")
    eval_dataset  = JsonlChatDataset(EVAL_FILE,  tokenizer, MAX_LENGTH)

    def collate_fn(batch):
        pad_id = tokenizer.pad_token_id
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            ids  = x["input_ids"] + [pad_id] * pad_len
            mask = x["attention_mask"] + [0] * pad_len
            lbl  = list(x["input_ids"])
            for i in range(min(x["prompt_len"], len(lbl))):
                lbl[i] = -100
            lbl = lbl + [-100] * pad_len
            input_ids.append(ids)
            attention_mask.append(mask)
            labels.append(lbl)
        return {
            "input_ids":      torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels":         torch.tensor(labels),
        }

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, collate_fn=collate_fn)
    eval_loader  = DataLoader(eval_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps   = (len(train_loader) * NUM_EPOCHS) // GRAD_ACCUM
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    global_step   = resume_step
    best_val_loss = resume_best_loss
    skip_target   = resume_step * GRAD_ACCUM
    micro_step    = 0

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"\n학습 시작! (총 {total_steps}스텝, warmup {warmup_steps}스텝, 재개 시작점 {resume_step}스텝)")

    train_start = time.time()
    step_times  = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            micro_step += 1
            if micro_step <= skip_target:
                continue

            step_start     = time.time()
            input_ids      = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            labels         = batch["labels"].to("cuda")

            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss / GRAD_ACCUM

            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                current_lr = get_cosine_lr(global_step, total_steps, warmup_steps, LR)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                elapsed = time.time() - step_start
                step_times.append(elapsed)
                if len(step_times) > 50:
                    step_times.pop(0)

                avg_sec     = sum(step_times) / len(step_times) * GRAD_ACCUM
                remain_step = total_steps - global_step
                eta_sec     = int(remain_step * avg_sec)
                eta_str     = f"{eta_sec//3600}h {(eta_sec%3600)//60}m {eta_sec%60}s"
                vram_used   = torch.cuda.memory_allocated() / 1024**3
                avg_loss = running_loss / GRAD_ACCUM
                line = (
                    f"\r[epoch {epoch+1}] step {global_step}/{total_steps} | "
                    f"loss {avg_loss:.4f} | lr {current_lr:.2e} | "
                    f"{avg_sec:.1f}s/step | ETA {eta_str} | "
                    f"VRAM {vram_used:.1f}GB"
                )
                print(line, end="", flush=True)
                running_loss = 0.0

                if global_step % EVAL_STEPS == 0:
                    print()
                    model.eval()
                    eval_loss = 0.0
                    with torch.no_grad():
                        for vbatch in eval_loader:
                            vi = vbatch["input_ids"].to("cuda")
                            vm = vbatch["attention_mask"].to("cuda")
                            vl = vbatch["labels"].to("cuda")
                            with autocast("cuda", dtype=torch.bfloat16):
                                vout = model(input_ids=vi, attention_mask=vm, labels=vl)
                            eval_loss += vout.loss.item()
                    eval_loss /= len(eval_loader)
                    print(f"  >> [평가] step {global_step} | eval_loss {eval_loss:.4f} | lr {current_lr:.2e}")
                    if eval_loss < best_val_loss:
                        best_val_loss = eval_loss
                        model.save_pretrained(OUTPUT_DIR)
                        tokenizer.save_pretrained(OUTPUT_DIR)
                        save_train_state(OUTPUT_DIR, global_step, best_val_loss)
                        print(f"  >> best 모델 저장 (eval_loss {eval_loss:.4f})")
                    model.train()

                if global_step % SAVE_STEPS == 0:
                    ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_dir)
                    save_train_state(ckpt_dir, global_step, best_val_loss)
                    print(f"  체크포인트 저장: {ckpt_dir}")

        print(f"\n[epoch {epoch+1}] 완료")

    print("\n학습 완료!")
    print(f"best val loss: {best_val_loss:.4f}")
    print(f"어댑터 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
