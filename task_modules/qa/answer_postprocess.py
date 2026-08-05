"""
QA answer postprocessing: chuẩn hoá câu trả lời.
Ví dụ: "5" → cả "5" và "Năm"; text → lowercase/upper tùy ngữ cảnh.
"""
import re
from typing import List

_VI_NUMBERS = {
    "0": "Không", "1": "Một", "2": "Hai", "3": "Ba", "4": "Bốn",
    "5": "Năm",   "6": "Sáu", "7": "Bảy", "8": "Tám", "9": "Chín",
    "10": "Mười",
}

_EN_NUMBERS = {
    "0": "zero", "1": "one", "2": "two",   "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten",
}


def normalize_answer(answer: str) -> List[str]:
    """
    Trả về danh sách các biểu diễn tương đương của câu trả lời.
    Ví dụ: "5" → ["5", "Năm", "five"]
    """
    answer = answer.strip()
    variants = {answer}

    # Nếu là số nguyên
    if re.fullmatch(r"\d+", answer):
        if answer in _VI_NUMBERS:
            variants.add(_VI_NUMBERS[answer])
        if answer in _EN_NUMBERS:
            variants.add(_EN_NUMBERS[answer])

    # Thêm lowercase/titlecase
    variants.add(answer.lower())
    variants.add(answer.capitalize())

    return list(variants)


def postprocess_qa_answer(answer: str, question: str = "") -> str:
    """
    Làm sạch câu trả lời: bỏ prefix thừa ("Câu trả lời: ", "Answer: "...).
    """
    answer = re.sub(r"^(Câu trả lời|Answer|Result)\s*[:：]\s*", "", answer, flags=re.IGNORECASE)
    answer = answer.strip().strip("\"'")
    return answer
