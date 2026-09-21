"""71859 평가셋과 71718/71716 학습 후보의 문제 본문 중복 검사.

id는 데이터셋마다 체계가 달라 겹침을 판단할 수 없다. 문제 본문을 정규화해 직접 비교한다.
정규화: LaTeX 기호·공백·구두점을 지우고 소문자화. 표기 차이(`$ x $` vs `$x$`)를 흡수한다.

왜 필요한가: 평가셋은 71859의 VL_ 폴더에서 왔다. 새 데이터를 학습에 넣었을 때
같은 문제가 학습과 평가에 동시에 들어 있으면 평가 점수가 부풀려진다.
"""
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

# 백슬래시가 들어간 LaTeX 명령을 먼저 지우고, 남은 기호를 지운다.
_LATEX_CMD = re.compile(r"\\(?:text|mathrm|left|right|mathbb|mathcal|displaystyle|quad|qquad)\b")
_PUNCT = re.compile(r"[\s$\\{}\[\]().,;:·]")


def norm(s: str) -> str:
    s = _LATEX_CMD.sub("", s or "")
    return _PUNCT.sub("", s).lower()


BROKEN: list[str] = []  # 읽지 못한 파일. 건수를 보고하기 위해 모아 둔다.


def q_of_71859(path: str) -> str:
    """71859: learning_data_info의 문항(텍스트). 깨진 파일은 건너뛰고 기록한다."""
    try:
        d = json.load(open(path, encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        BROKEN.append(path)
        return ""
    out = []
    for it in d.get("learning_data_info", []):
        if "문항(텍스트)" in it.get("class_name", ""):
            for info in it.get("class_info_list", []):
                t = (info.get("text_description") or "").strip()
                if t:
                    out.append(t)
    return " ".join(out)


def walk_json(root: str):
    for dp, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".json"):
                yield os.path.join(dp, f)


GRADES = ["초등학교_3학년", "초등학교_4학년", "초등학교_5학년", "초등학교_6학년",
          "중학교_1학년", "중학교_2학년", "중학교_3학년", "고등학교_공통수학"]


def load_71859(prefix: str) -> dict[str, str]:
    base = "D:/수학교과문제풀이데이터"
    out = {}
    for d in os.listdir(base):
        if d.startswith(prefix):
            for p in walk_json(f"{base}/{d}"):
                q = q_of_71859(p)
                if q:
                    out[norm(q)] = p
    return out


def scan_71718(eval_q: dict, train_q: dict) -> None:
    b = "D:/수학 과목 문제생성 데이터"
    n = hit_eval = hit_train = 0
    for g in GRADES:
        d = f"{b}/TL_1.문제_{g}"
        for nm in os.listdir(d):
            j = json.load(open(f"{d}/{nm}", encoding="utf-8-sig"))
            qt = norm(j["OCR_info"][0].get("question_text") or "")
            if len(qt) < 8:
                continue
            n += 1
            hit_eval += qt in eval_q
            hit_train += qt in train_q
    print(f"71718 학습 문항 {n}건")
    print(f"  71859 평가 문항과 동일: {hit_eval}건 ({hit_eval / n:.2%})")
    print(f"  71859 학습 문항과 동일: {hit_train}건 ({hit_train / n:.2%})")


def near_duplicates(thresholds=(0.95, 0.9, 0.8)) -> None:
    """유사 문항 검사. 숫자만 바꾼 변형 문제는 정확 일치로 안 잡힌다.

    문자 3-gram TF-IDF 코사인으로 평가 문항마다 71718 학습 문항 중 가장 가까운 것을 찾는다.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    eval_q = list(load_71859("VL_").keys())
    b = "D:/수학 과목 문제생성 데이터"
    pool = []
    for g in GRADES:
        d = f"{b}/TL_1.문제_{g}"
        for nm in os.listdir(d):
            j = json.load(open(f"{d}/{nm}", encoding="utf-8-sig"))
            qt = norm(j["OCR_info"][0].get("question_text") or "")
            if len(qt) >= 8:
                pool.append(qt)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2, dtype="float32")
    P = vec.fit_transform(pool)
    E = vec.transform(eval_q)
    best = []
    step = 200
    for i in range(0, E.shape[0], step):
        sim = E[i:i + step] @ P.T
        best.extend(sim.max(axis=1).toarray().ravel().tolist())
    n = len(best)
    print(f"평가 문항 {n}건 vs 71718 학습 문항 {len(pool)}건 (문자 3-gram TF-IDF 코사인, 최근접)")
    for t in thresholds:
        c = sum(1 for s in best if s >= t)
        print(f"  유사도 >= {t}: {c}건 ({c / n:.1%})")


def write_exclusion(out_path: str, threshold: float = 0.9) -> None:
    """71859 평가 문항과 비슷한 71718 학습 문항의 id 목록을 쓴다.

    near_duplicates()는 '평가 문항마다 가장 가까운 학습 문항'을 찾는다. 학습에서 빼야
    할 것은 그 반대 방향, 즉 '평가 문항과 비슷한 학습 문항'이라서 같은 유사도 행렬을
    학습 문항 기준으로 다시 훑는다. 평가 문항 하나가 학습 문항 여러 개와 비슷할 수 있다.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    eval_q = list(load_71859("VL_").keys())
    b = "D:/수학 과목 문제생성 데이터"
    pool, ids = [], []
    for g in GRADES:
        d = f"{b}/TL_1.문제_{g}"
        for nm in os.listdir(d):
            j = json.load(open(f"{d}/{nm}", encoding="utf-8-sig"))
            qt = norm(j["OCR_info"][0].get("question_text") or "")
            if len(qt) >= 8:
                pool.append(qt)
                ids.append(j["id"])
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2, dtype="float32")
    P = vec.fit_transform(pool)
    E = vec.transform(eval_q)
    ET = E.T.tocsc()
    flagged = []
    step = 4000
    for i in range(0, P.shape[0], step):
        mx = (P[i:i + step] @ ET).max(axis=1).toarray().ravel()
        flagged.extend(ids[i + k] for k, v in enumerate(mx) if v >= threshold)
    json.dump(sorted(set(flagged)), open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"71718 학습 {len(pool)}건 중 71859 평가 문항과 유사도 >= {threshold}: "
          f"{len(set(flagged))}건 -> {out_path}")


def main() -> None:
    if "--write-exclusion" in sys.argv:
        write_exclusion(sys.argv[sys.argv.index("--write-exclusion") + 1])
        return
    eval_q = load_71859("VL_")
    train_q = load_71859("TL_")
    print(f"71859 읽지 못한 파일: {len(BROKEN)}건" + (f" (예: {BROKEN[0]})" if BROKEN else ""))
    print(f"71859 평가 문항(고유) {len(eval_q)} / 학습 문항(고유) {len(train_q)}")
    print(f"71859 내부 학습-평가 중복: {len(set(eval_q) & set(train_q))}건")
    scan_71718(eval_q, train_q)
    if "--near" in sys.argv:
        near_duplicates()


if __name__ == "__main__":
    main()
