"""
Pipeline 1: Memory Extraction
단기 대화 로그 → LLM 기반 기억 추출 → Supabase REST API로 벡터 저장.
"""

import json
import logging
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.database import (
    check_duplicate_memory,
    get_unextracted_conversations,
    mark_as_extracted,
    save_long_term_memory,
)
from app.schemas import ExtractionResult

logger = logging.getLogger(__name__)

# 한 번에 너무 많은 단기 로그를 넣으면 LLM 출력 JSON이 잘리거나 깨질 수 있어 청크로 나눈다.
EXTRACTION_CHUNK_SIZE = 40

EXTRACTION_SYSTEM_PROMPT = """# Role
당신은 60대 이상 독거 어르신의 대화 이력을 분석하여, 장기 기억으로 저장할 가치가 있는 핵심 정보를 추출하는 '시니어 특화 데이터 추출 에이전트'입니다.

# Objective
제공된 대화 로그(Conversation Logs)를 분석하여 어르신의 지속적인 선호도, 일상 루틴, 가족 관계, 중요한 사건 및 건강/감정 상태를 지정된 JSON 형식으로 완벽하게 추출하세요.

# Extraction Guidelines
1. memory_date (날짜): 타임스탬프 참고하여 절대적 날짜(YYYY-MM-DD) 추론.
2. content (기억 내용): 3인칭 관찰자 시점 명료한 단문 요약.
3. category (카테고리): [Family, Health, Routine, Preference, Event, Emotion] 중 택 1.
4. emotion (감정): 주된 감정 (기쁨, 슬픔, 외로움, 아쉬움 등).
5. importance_score (중요도 점수): 1점(낮음/일상) ~ 10점(높음/경조사, 심각한 질환, 우울감).
6. entity (주요 대상): 주요 인물, 사물, 장소 배열(Array) 형태.

# Output Format (반드시 JSON만 출력)
{
  "memory": [
    {
      "memory_date": "YYYY-MM-DD",
      "content": "string",
      "category": "string",
      "emotion": "string",
      "importance_score": integer,
      "entity": ["string"]
    }
  ]
}
"""


def _build_conversation_text(conversations: list[dict]) -> str:
    lines: list[str] = []
    for msg in conversations:
        ts = msg.get("chat_time", "")
        if isinstance(ts, str) and "T" in ts:
            ts = ts[:16].replace("T", " ")
        lines.append(f"[{ts}] {msg['role']}: {msg['content']}")
    return "\n".join(lines)


async def extract_memories(
    user_id: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """미추출 대화에서 기억을 추출하고 벡터 저장한다."""
    settings = get_settings()

    conversations = get_unextracted_conversations(user_id=user_id, limit=limit)
    if not conversations:
        logger.info("추출할 미처리 대화가 없습니다.")
        return []

    grouped: dict[str, list] = {}
    for msg in conversations:
        grouped.setdefault(msg["user_id"], []).append(msg)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        project=settings.gcp_project_id,
        location=settings.gcp_location,
        temperature=0.1,
        max_output_tokens=8192,
        thinking_budget=0,
    )
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        project=settings.gcp_project_id,
        location=settings.gcp_location,
    )

    all_extracted: list[dict] = []

    for uid, msgs in grouped.items():
        for start in range(0, len(msgs), EXTRACTION_CHUNK_SIZE):
            chunk = msgs[start : start + EXTRACTION_CHUNK_SIZE]
            conv_text = _build_conversation_text(chunk)
            prompt_text = (
                f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
                f"# Conversation Logs\n{conv_text}"
            )

            try:
                response = await llm.ainvoke(prompt_text)
                raw = response.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                parsed = ExtractionResult.model_validate(json.loads(raw))
            except Exception:
                logger.exception(
                    "LLM 기억 추출 실패 (user_id=%s, chunk=%d~%d)",
                    uid,
                    start + 1,
                    start + len(chunk),
                )
                continue

            for mem in parsed.memory:
                try:
                    emb = embeddings_model.embed_query(mem.content)

                    if check_duplicate_memory(uid, mem.content):
                        logger.info("중복 기억 스킵: user_id=%s, content=%s", uid, mem.content[:40])
                        continue

                    save_long_term_memory(
                        user_id=uid,
                        memory_date=mem.memory_date,
                        content=mem.content,
                        embedding=emb,
                        metadata={
                            "category": mem.category,
                            "emotion": mem.emotion,
                            "importance_score": mem.importance_score,
                            "entity": mem.entity,
                        },
                    )
                    all_extracted.append(mem.model_dump())
                except Exception:
                    logger.exception("기억 저장 실패 (user_id=%s, content=%s)", uid, mem.content[:30])
                    continue

            chat_ids = [m["chat_id"] for m in chunk]
            mark_as_extracted(chat_ids)
            logger.info(
                "user_id=%s: %d건 대화 청크에서 %d건 기억 추출 완료",
                uid,
                len(chunk),
                len(parsed.memory),
            )

    return all_extracted
