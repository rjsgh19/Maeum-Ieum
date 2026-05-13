"""
Pipeline 2: Empathetic RAG Chat
사용자 발화 → 쿼리 임베딩 → 하이브리드 검색(벡터 + GIN 메타데이터 via Supabase RPC) → 공감형 답변 생성.

응답 지연을 줄이기 위해 검색 전 LLM 호출(키워드 추출)은 하지 않고, 경량 휴리스틱만 사용한다.
장기기억 추출은 API 레이어에서 BackgroundTasks로 처리한다.
"""

import asyncio
import json
import logging
import re
from difflib import SequenceMatcher

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.database import HybridSearchResult, hybrid_search

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """# Role
당신은 노인을 위한 정서지원 대화 코치이자, 따뜻한 말벗입니다. 답변은 사람 냄새가 나고 따뜻해야 하며, 사용자의 지난 이야기(장기기억)를 섬세하게 연결하는 훌륭한 동반자입니다.

# 대화 상대 전용 기억 (절대 준수)
아래 [장기기억]은 오직 이 대화 상대(동일 사용자)에게서만 수집된 내용입니다. 다른 분의 이야기로 착각하지 마세요. 기억에 없는 사실은 지어내지 말고, 필요하면 부드럽게 되물어보세요.

{user_context}

# Objective
- 아래 사용자 발화와 [장기기억]을 바탕으로, 노인을 배려하는 공감형 맞춤 답변 1개를 만든다.
- 따뜻하고 존중감 있으며, 이해하기 쉬운 일상적인 한국어(해요체 존댓말)로 쓴다.
- 외로움, 불안, 서운함, 상실감, 건강 걱정, 가족 문제가 드러나면 먼저 감정을 수용하고 정서적 안전 기지를 제공한다.

# Core Principles (절대 원칙)
1. [감정 수용] 먼저 사용자의 감정을 알아차리고 자연스럽게 짚어준다.
2. [짧은 공감] 훈계, 평가, 교정, 진단은 하지 않으며, 부드럽고 짧게 공감한다.
3. [최소한의 제안] 필요할 때만 아주 작고 실천 가능한 제안이나 위로를 하나만 덧붙인다.
4. [대화의 여백] 부담 없이 다음 말을 이어갈 수 있도록 열린 맺음말을 쓴다.
5. [분량 제한] 최종 답변(response)은 2~4문장으로, 장황하지 않게 쓴다.
6. [존중의 태도] 어린아이 대하듯 하지 않고(비종속적), 자연스럽고 정중한 존댓말을 유지한다.
7. [해결책 지양] 위험 신호가 없는 일반 대화에서는 의사·상담사·가족 방문 등 해결책 나열보다, 지금 여기의 공감을 우선한다.

# 장기기억 활용 가이드
- 가족, 지병, 취미, 최근 일 등 기억을 대화에 자연스럽게 녹인다.
- 기계적으로 기억을 과시하지 않는다. ("제 기억 데이터에 따르면", "예전에 말씀하셨듯이" 등 AI 티 나는 표현 금지)
- 맥락상 적절할 때만 가볍고 다정하게 반영하고, 매번 억지로 끌어오지 않는다.

# 말투 및 톤앤매너
- 권장: "많이 속상하셨겠어요", "그럴 때 마음이 헛헛하실 수 있지요", "혼자 견디시기 버거우셨겠어요" 등 차분·온화한 어미.
- 금지: "그건 잘못된 생각입니다", "원래 다 그런 겁니다", "긍정적으로 생각하세요", "파이팅!" 등 감정을 차단하거나 가벼운 유행어.
- 과장 금지: "제가 다 이해합니다", "저라도 그랬을 겁니다" 같은 과한 공감은 피한다.
- 쉬운 말: 불필요한 외래어·젊은 층 은어·어려운 한자어는 줄이고 직관적인 우리말을 쓴다.

# 가드레일 및 안전
- 자해, 극단적 선택, 학대, 심각한 응급(생명·안전) 징후가 보이면 일반 공감 모드를 멈추고 안전 우선 응답(response)을 쓴다.
- 안전 우선일 때 response에서는: (1) 두려움·고통에 즉시 공감 (2) 혼자 있지 말 것을 다정하지만 단호하게 권함 (3) 가족·보호자·119(응급)·지역 정신건강복지센터(☎ 1577-0199) 등 구체적 도움 연락을 권유 (4) 의료·법률 진단은 내리지 않음.
- 이때 alert 필드는 반드시 "DEPRESSION"으로 둔다(극심한 우울·자해·극단적 언급 포함). 격한 분노·폭력 위협이 중심이면 alert는 "ANGER". 해당 없으면 alert는 null.

# 상황별 맞춤
- 외로움: 존재 가치를 인정하고 연결감을 느낄 수 있는 표현.
- 건강 불안: 섣불리 안심시키거나 겁주지 말고, 걱정되는 마음을 받아준다.
- 가족 서운함: 누구 잘잘못을 따지지 않고 서운한 마음만 온전히 듣는다.
- 과거 회상: 지나온 삶과 기억을 소중히 여기며 존경이 담기게 반응한다.

# 출력 형식 (엄수)
- 오직 JSON 한 덩어리만 출력한다. 마크다운 코드펜스(```), 설명 문장, 인사말을 절대 붙이지 않는다.
- 필드:
  - "alert": null 또는 문자열 "DEPRESSION" 또는 "ANGER" (위 안전 규칙에 따름)
  - "emotion": 사용자 발화의 핵심 감정을 짧게(한두 가지 느낌을 쉼표로 묶어도 됨)
  - "intent": 사용자가 바라는 정서적 욕구나 의도를 한 문장 이내로
  - "response": 장기기억이 자연스럽게 반영된 최종 답변 본문(2~4문장, 존댓말)

[장기기억]
{context}
"""

ALERT_PATTERN = re.compile(r"^\[ALERT:\s*(\w+)\]\s*")
EMO_PATTERN = re.compile(r"\[EMO:\s*(\S+?)\]\s*$")
FALLBACK_ANSWER = "어르신, 제가 잘 알아듣지 못했어요. 다시 한번 말씀해 주시겠어요?"

_AVATAR_PERSONA = {
    "boy": "화면의 말벗은 다정한 소년 손주 캐릭터입니다. 말투는 밝고 순수하게 유지하세요.",
    "girl": "화면의 말벗은 다정한 소녀 손주 캐릭터입니다. 말투는 부드럽고 따뜻하게 유지하세요.",
    "dog": "화면의 말벗은 귀여운 강아지 캐릭터입니다. 짧고 활기 있는 문장으로 반응하되 존댓말은 유지하세요.",
    "cat": "화면의 말벗은 차분한 고양이 캐릭터입니다. 여유 있고 담백한 톤으로 말하되 존댓말은 유지하세요.",
    "robot": "화면의 말벗은 친절한 로봇 캐릭터입니다. 정중하고 명확한 말투로 답하되 딱딱하지 않게 하세요.",
}


def _format_user_context(
    display_name: str | None,
    gender: str | None,
    age: int | None,
    avatar_id: str | None,
) -> str:
    """프론트에서 넘긴 프로필을 시스템 프롬프트에 삽입할 블록으로 만든다."""
    lines: list[str] = []
    if display_name and display_name.strip():
        name = display_name.strip()
        lines.append(f"- 어르신 이름(호칭): '{name} 님'이라고 부르세요.")
    if gender and gender.strip():
        lines.append(
            f"- 성별 정보: {gender.strip()}. 호칭은 상황에 맞게 할머니/할아버지/어르신 등 자연스럽게 선택하세요."
        )
    if age is not None and 1 <= age <= 120:
        lines.append(f"- 나이: {age}세.")
    if avatar_id and avatar_id in _AVATAR_PERSONA:
        lines.append(f"- {_AVATAR_PERSONA[avatar_id]}")
    if not lines:
        return ""
    return "[사용자 프로필 — 답변에 자연스럽게 반영]\n" + "\n".join(lines)


def _get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        project=settings.gcp_project_id,
        location=settings.gcp_location,
    )


def _quick_entities_from_message(message: str) -> list[str] | None:
    """
    LLM 없이 하이브리드 검색의 메타데이터(entity) 힌트만 채운다.
    검색 전 추가 왕복을 없애 응답을 앞당긴다.
    """
    if not message or not message.strip():
        return None
    parts = re.findall(r"[가-힣]{2,}|[A-Za-z]{3,}", message)
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= 5:
            break
    return out or None


def _deduplicate(results: list[HybridSearchResult], similarity_threshold: float = 0.85) -> list[HybridSearchResult]:
    """텍스트가 거의 동일한 기억만 제거한다 (threshold=0.85 → 내용이 다른 기억은 보존)."""
    kept: list[HybridSearchResult] = []
    for r in results:
        is_dup = False
        for k in kept:
            ratio = SequenceMatcher(None, r.content, k.content).ratio()
            if ratio >= similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(r)
    return kept


def _format_memories(results: list[HybridSearchResult]) -> str:
    if not results:
        return "(아직 참고할 기억이 없습니다. 어르신의 말씀에 자연스럽게 반응해 주세요.)"
    lines: list[str] = []
    for r in results:
        meta = r.metadata or {}
        imp = meta.get("importance_score", 5)
        tag_parts = [meta.get("category", ""), f"감정:{meta.get('emotion', '')}"]
        if meta.get("entity"):
            tag_parts.append(f"관련:{','.join(meta['entity'])}")
        if int(imp) >= 7:
            tag_parts.append(f"★중요도:{imp}")
        tag = " | ".join(p for p in tag_parts if p)
        lines.append(f"- [{tag}] {r.content}")
    return "\n".join(lines)


def _parse_alert(text: str) -> tuple[str, str | None]:
    """답변 텍스트에서 [ALERT: ...] 플래그를 분리한다."""
    match = ALERT_PATTERN.match(text)
    if match:
        alert_type = match.group(1)
        clean_text = text[match.end():].strip()
        return clean_text, alert_type
    return text, None


def _parse_emotion_tag(text: str) -> tuple[str, str]:
    """답변 끝의 [EMO:감정] 태그를 분리한다. 없으면 '중립' 반환."""
    match = EMO_PATTERN.search(text)
    if match:
        emotion = match.group(1)
        clean_text = text[:match.start()].strip()
        return clean_text, emotion
    return text, "중립"


def _strip_json_fence(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        s = s.rsplit("```", 1)[0].strip()
    return s


def _coerce_emotion_for_care(emotion_field: str) -> str:
    """정서 엔진 호환용: JSON emotion 문구를 기존 태그 집합에 가깝게 매핑한다."""
    if not emotion_field or not emotion_field.strip():
        return "중립"
    s = emotion_field.strip()
    checks = [
        ("슬픔", ["슬픔", "슬프", "서운", "서러", "우울", "허전", "외로", "상실", "아쉽", "눈물"]),
        ("분노", ["분노", "화", "짜증", "억울", "미워", "격분"]),
        ("공포", ["공포", "불안", "무서", "두려", "조마", "초조"]),
        ("기쁨", ["기쁨", "행복", "즐거", "신나", "고마", "감사", "편안", "뿌듯"]),
        ("놀람", ["놀람", "놀랐", "깜짝"]),
        ("혐오", ["혐오", "구역질", "역겨"]),
    ]
    for tag, needles in checks:
        for n in needles:
            if n in s:
                return tag
    return "중립"


def _parse_chat_json_output(raw: str) -> tuple[str, str | None, str] | None:
    """JSON 모드 파싱. 성공 시 (answer, alert, detected_emotion_for_care)."""
    try:
        s = _strip_json_fence(raw)
        obj = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    response = obj.get("response")
    if not isinstance(response, str) or not response.strip():
        return None
    alert_raw = obj.get("alert")
    alert: str | None
    if alert_raw is None or alert_raw == "":
        alert = None
    elif isinstance(alert_raw, str):
        a = alert_raw.strip().upper()
        alert = a if a in ("DEPRESSION", "ANGER") else None
    else:
        alert = None
    emotion_field = obj.get("emotion", "")
    detected = _coerce_emotion_for_care(str(emotion_field) if emotion_field is not None else "")
    return (response.strip(), alert, detected)


def _parse_llm_chat_output(raw: str) -> tuple[str, str | None, str]:
    """JSON 우선, 실패 시 구식 [ALERT]/[EMO] 태그 파서로 폴백."""
    parsed = _parse_chat_json_output(raw)
    if parsed is not None:
        return parsed
    answer, alert = _parse_alert(raw)
    answer, detected_emotion = _parse_emotion_tag(answer)
    return answer, alert, detected_emotion


async def empathetic_chat(
    user_id: str,
    message: str,
    *,
    category: str | None = None,
    emotion: str | None = None,
    display_name: str | None = None,
    gender: str | None = None,
    age: int | None = None,
    avatar_id: str | None = None,
    fetch_k: int = 15,
    final_k: int = 5,
    vector_weight: float = 0.65,
    metadata_weight: float = 0.10,
    recency_weight: float = 0.15,
    importance_weight: float = 0.10,
) -> dict:
    """하이브리드 RAG 기반 공감형 챗봇 응답을 생성한다.

    장기기억 검색·컨텍스트는 항상 ``user_id``로 격리된다 (Supabase ``hybrid_search``의
    ``WHERE user_id = p_user_id``). 다른 사용자의 벡터/메타데이터는 후보에 포함되지 않는다.
    """
    settings = get_settings()

    emb_model = _get_embeddings_model()
    query_embedding = await emb_model.aembed_query(message)

    query_entities = _quick_entities_from_message(message)

    # 동기 Supabase RPC — 이벤트 루프 블로킹 방지
    raw_results = await asyncio.to_thread(
        hybrid_search,
        user_id,
        query_embedding,
        category=category,
        emotion=emotion,
        entities=query_entities,
        top_k=fetch_k,
        vector_weight=vector_weight,
        metadata_weight=metadata_weight,
        recency_weight=recency_weight,
        importance_weight=importance_weight,
    )

    results = _deduplicate(raw_results)[:final_k]

    context_str = _format_memories(results)
    logger.info(
        "하이브리드 검색 완료 (user_id 격리): user=%s, 원본=%d건→중복제거=%d건, category=%s, emotion=%s",
        user_id, len(raw_results), len(results), category, emotion,
    )

    user_context = _format_user_context(display_name, gender, age, avatar_id)

    prompt = ChatPromptTemplate.from_messages([
        ("system", CHAT_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        project=settings.gcp_project_id,
        location=settings.gcp_location,
        temperature=0.7,
        max_output_tokens=512,
        thinking_budget=0,
    )

    formatted = prompt.invoke(
        {"context": context_str, "question": message, "user_context": user_context}
    )
    resp = await llm.ainvoke(formatted.messages)
    raw_answer = resp.content or ""

    if not raw_answer.strip():
        logger.warning("LLM 빈 응답 — fallback 사용")
        raw_answer = FALLBACK_ANSWER

    answer, alert, detected_emotion = _parse_llm_chat_output(raw_answer)
    if not answer.strip():
        logger.warning("LLM 파싱 실패 또는 빈 response — fallback 사용")
        answer = FALLBACK_ANSWER
        alert = None
        detected_emotion = "중립"

    return {
        "answer": answer,
        "alert": alert,
        "detected_emotion": detected_emotion,
        "retrieved_memories": [r.content for r in results],
    }
