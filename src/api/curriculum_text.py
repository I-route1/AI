import re
from html import unescape


def _clean_text(text: str) -> str:
    """교과 지문의 HTML만 정리한다. 수식·부등호·구간과 빈칸은 문제의 일부다."""
    text = re.sub(r'<sup>([^<>]+)</sup>', r'^{\1}', text, flags=re.I)
    text = re.sub(r'<sub>([^<>]+)</sub>', r'_{\1}', text, flags=re.I)
    text = re.sub(r'</?(?:p|br|div|span|b|i|em|strong|li|ul|ol|table|tr|td|th)\b[^>]*>',
                  ' ', text, flags=re.I)
    return re.sub(r'\s+', ' ', unescape(text)).strip()


def extract_passage(doc: str) -> str:
    """
    RAG 문서에서 핵심 내용만 추출 — 폴더 경로·과정 코드 제거.
    우선순위: 학습 지문 → Q&A 답변 → 성취기준 텍스트 → 전체 클리닝
    """
    # 학습 지문
    boundary = r'(?=\n\s*(?:\[(?:대상|학습 지문|교육과정 성취기준|관련 Q&A|질문|답변|문제|정답|해설)\]|질문[:：])|$)'
    m = re.search(r'\[학습 지문\]\s*(.+?)' + boundary, doc, re.DOTALL)
    if m:
        return _clean_text(m.group(1).strip())

    # Q&A 답변
    m = re.search(r'답변[：:\s]+(.+?)' + boundary, doc, re.DOTALL)
    if m:
        return _clean_text(m.group(1).strip())

    # 성취기준 설명 (코드 번호 제외)
    m = re.search(r'\[교육과정 성취기준\]\s*\[[^\]]*\]\s*(.+?)' + boundary, doc, re.DOTALL)
    if m:
        return _clean_text(m.group(1).strip())

    return _clean_text(doc)
