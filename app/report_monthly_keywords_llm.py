"""
보호자 리포트 '이번 달 대화 키워드': short_term_history user 발화 → Gemini로 주제 키워드 추출.
실패 시 호출부에서 빈도 기반 fallback 사용.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings

logger = logging.getLogger(__name__)

_MAX_INPUT_CHARS = 12_000
_MAX_LINE_CHARS = 200
_PROMPT = """당신은 돌봄·노인 상담 보고서를 돕는 분석가입니다.
아래는 한 분이 이번 달에 남긴 **사용자(어르신) 발화** 일부입니다. (최신 쪽이 앞에 가깝게 섞여 있을 수 있습니다.)

과제: 보호자가 한눈에 이해할 수 있도록, 이번 달 대화에서 **반복되거나 중심이 된 주제·관심사**를 나타내는 **한국어 키워드**만 3~5개 뽑아 주세요.

규칙:
- 각 키워드는 **2~8자**의 **명사 또는 명사구**(예: 가족, 산책, 아프다, 병원, 손주) 위주. 조사·어미만 붙은 말은 피하세요.
- 인사·감탄·의미 없는 반복(안녕, 네, 응, 그래 등)은 키워드에서 제외하세요.
- 민감한 개인정보(실명·주소·전화번호 등)는 키워드에 넣지 마세요. 일반화된 표현만 사용하세요.
- 키워드끼리 **의미가 겹치지 않게** 다양하게 고르세요.

반드시 아래 JSON만 출력하세요 (코드 블록·설명 금지):
{{"keywords": ["첫째키워드", "둘째키워드", ...]}}
"keywords" 배열 길이는 3 이상 5 이하를 권장합니다. 발화가 매우 적으면 그만큼만(최소 1개) 넣어도 됩니다.

--- 발화 ---
{utterances_block}
"""


def _build_utterances_block(texts: list[str]) -> str:
    lines: list[str] = []
    used = 0
    for i, raw in enumerate(texts, 1):
        t = (raw or "").strip().replace("\n", " ")
        if not t:
            continue
        if len(t) > _MAX_LINE_CHARS:
            t = t[: _MAX_LINE_CHARS] + "…"
        piece = f"{i}. {t}"
        if used + len(piece) + 1 > _MAX_INPUT_CHARS:
            break
        lines.append(piece)
        used += len(piece) + 1
    return "\n".join(lines)


_WORD_OK = re.compile(r"^[가-힣]{2,8}$")


def _normalize_keywords(raw_list: list) -> list[dict]:
    words: list[str] = []
    seen: set[str] = set()
    for item in raw_list:
        if isinstance(item, dict):
            w = str(item.get("word") or item.get("keyword") or "").strip()
        else:
            w = str(item).strip()
        if not w or not _WORD_OK.match(w):
            continue
        if w in seen:
            continue
        seen.add(w)
        words.append(w)
        if len(words) >= 5:
            break
    n = len(words)
    return [{"word": w, "count": n - i} for i, w in enumerate(words)]


async def extract_monthly_keywords_with_llm(
    texts: list[str],
    *,
    top_n: int = 5,
) -> list[dict]:
    """
    DB에서 가져온 사용자 발화 문자열 목록으로 월간 키워드 추출.
    성공 시 [{{"word", "count"}}, ...] (최대 top_n), 실패 시 빈 리스트.
    """
    if not texts:
        return []

    block = _build_utterances_block(texts)
    if len(block.strip()) < 20:
        return []

    prompt = _PROMPT.format(utterances_block=block)

    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        project=settings.gcp_project_id,
        location=settings.gcp_location,
        temperature=0.35,
        max_output_tokens=256,
        thinking_budget=0,
    )

    try:
        resp = await llm.ainvoke(prompt)
        raw = (resp.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
        kws = data.get("keywords")
        if not isinstance(kws, list):
            logger.warning("월간 키워드 LLM: keywords가 배열이 아님")
            return []
        normalized = _normalize_keywords(kws)
        if not normalized:
            return []
        return normalized[:top_n]
    except Exception:
        logger.exception("월간 키워드 LLM 추출 실패")
        return []
