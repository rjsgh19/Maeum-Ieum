"""
리포트 '이달의 따뜻한 기억' 멘트: long_term_memory content → Gemini로 자연스러운 한 문장 생성.
챗봇과 동일하게 ChatGoogleGenerativeAI(gemini-2.5-flash) 사용.
"""

from __future__ import annotations

import json
import logging
from datetime import date

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings

logger = logging.getLogger(__name__)

_POSITIVE_EMOTIONS = {"행복", "기쁨", "감사", "사랑", "희망", "만족", "평온", "즐거움", "설렘"}

_EMOTION_SUFFIXES = {
    "행복": "정말 행복해 보이셨어요.",
    "기쁨": "목소리가 밝으셨어요.",
    "감사": "감사한 마음이 느껴졌어요.",
    "사랑": "사랑이 듬뿍 느껴졌어요.",
    "희망": "희망찬 모습이셨어요.",
    "즐거움": "정말 즐거워 보이셨어요.",
}

_BATCH_PROMPT = """당신은 노인 회상 요법(Reminiscence Therapy) 보조 역할입니다.
아래는 한 분의 대화에서 추출된 긍정적 장기기억 목록입니다. 각 항목마다 **한 문장**씩, 어르신께 들려주기 좋은 따뜻한 말을 써 주세요.

규칙:
- 반드시 존댓말(해요체).
- 문장은 짧고 자연스럽게(대략 50자 이내). 항목마다 표현을 달리하세요.
- 기억의 핵심(누구와, 무엇을, 어떤 기분)을 살려 다시 말하되, 원문을 그대로 복사하지 마세요.
- 의학적 진단·단정은 하지 마세요.
- 날짜가 있으면 "○월 ○일쯤"처럼 자연스럽게 넣어도 좋습니다.

항목:
{items_block}

반드시 아래 JSON만 출력하세요 (코드 블록·설명 금지):
{{"lines": ["첫 번째 문장", "두 번째 문장", ...]}}
"lines" 배열 길이는 항목 개수와 **정확히 같아야** 합니다.
"""


def _fallback_message(content: str, mem_date: str | None, emotion: str) -> str:
    short = (content or "")[:50].rstrip(".")
    date_prefix = ""
    if mem_date:
        try:
            d = date.fromisoformat(str(mem_date))
            date_prefix = f"{d.month}월 {d.day}일, "
        except (ValueError, TypeError):
            pass
    suffix = _EMOTION_SUFFIXES.get(emotion, "목소리가 밝으셨어요.")
    return f"{date_prefix}'{short}' 이야기를 하실 때 {suffix}"


async def generate_reminiscence_lines(candidates: list[dict]) -> list[str]:
    """
    candidates: 각 dict는 content, memory_date(optional), emotion(optional str).
    반환: 동일 순서의 한국어 멘트 문자열 리스트.
    """
    if not candidates:
        return []

    n = len(candidates)
    settings = get_settings()
    lines_block: list[str] = []
    for i, c in enumerate(candidates, 1):
        d = c.get("memory_date") or ""
        em = c.get("emotion") or ""
        body = (c.get("content") or "").strip().replace("\n", " ")
        if len(body) > 280:
            body = body[:280] + "…"
        lines_block.append(f"{i}. 날짜: {d}, 감정: {em}, 기억: {body}")

    prompt = _BATCH_PROMPT.format(items_block="\n".join(lines_block))

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        project=settings.gcp_project_id,
        location=settings.gcp_location,
        temperature=0.65,
        max_output_tokens=400,
        thinking_budget=0,
    )

    try:
        resp = await llm.ainvoke(prompt)
        raw = (resp.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        lines = data.get("lines")
        if isinstance(lines, list) and len(lines) == n:
            out = [str(x).strip() for x in lines]
            if all(out):
                return out
        logger.warning(
            "회상 LLM 응답 형식 이상: 기대 줄 수=%d, 실제=%s",
            n,
            type(lines).__name__ if lines is not None else "None",
        )
    except Exception:
        logger.exception("회상 멘트 LLM 생성 실패 — 템플릿으로 대체")

    return [
        _fallback_message(
            c.get("content") or "",
            c.get("memory_date"),
            str(c.get("emotion") or ""),
        )
        for c in candidates
    ]


def filter_positive_memory_candidates(memories: list[dict], max_items: int = 3) -> list[dict]:
    """long_term_memory 행 목록에서 긍정/고중요도 후보만 고른다."""
    candidates: list[dict] = []
    for m in memories:
        meta = m.get("metadata") or {}
        emotion = meta.get("emotion", "")
        if isinstance(emotion, str):
            emotion = emotion.strip()
        importance = meta.get("importance", 0)
        if emotion in _POSITIVE_EMOTIONS or (
            isinstance(importance, (int, float)) and importance >= 8
        ):
            candidates.append(
                {
                    "content": m.get("content", ""),
                    "memory_date": m.get("memory_date"),
                    "emotion": emotion,
                }
            )
            if len(candidates) >= max_items:
                break
    return candidates
