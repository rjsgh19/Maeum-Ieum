"""
Supabase REST API (PostgREST) 기반 DB 클라이언트.
IPv6 전용 DB에서도 문제없이 동작한다.
"""

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

from supabase import Client, create_client

from app.config import get_settings

logger = logging.getLogger(__name__)

_client: Client | None = None


def _disable_proxy_env_for_backend() -> None:
    """
    로컬 프록시(HTTP_PROXY/HTTPS_PROXY)가 켜져 있으면
    Supabase PostgREST 요청이 403 ProxyError로 실패할 수 있어 비활성화한다.
    """
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(key, None)


def get_supabase() -> Client:
    global _client
    if _client is None:
        _disable_proxy_env_for_backend()
        settings = get_settings()
        _client = create_client(settings.supabase_url, settings.supabase_key)
    return _client


# ── Short-term History CRUD ──────────────────────────────────────────

def save_chat_message(user_id: str, role: str, content: str) -> dict:
    sb = get_supabase()
    row = {
        "user_id": user_id,
        "role": role,
        "content": content,
    }
    result = sb.table("short_term_history").insert(row).execute()
    return result.data[0] if result.data else row


def get_unextracted_conversations(
    user_id: str | None = None,
    limit: int = 50,
) -> list[dict]:
    sb = get_supabase()
    query = (
        sb.table("short_term_history")
        .select("*")
        .eq("is_extracted", False)
        .order("chat_time", desc=False)
        .limit(limit)
    )
    if user_id:
        query = query.eq("user_id", user_id)
    result = query.execute()
    return result.data or []


def mark_as_extracted(chat_ids: list[str]) -> None:
    sb = get_supabase()
    sb.table("short_term_history").update(
        {"is_extracted": True}
    ).in_("chat_id", chat_ids).execute()


def get_user_messages_since(
    user_id: str,
    since_iso: str,
    *,
    limit: int = 500,
) -> list[str]:
    """user 역할 메시지 중 chat_time >= since_iso 인 content 목록 (최신순)."""
    sb = get_supabase()
    result = (
        sb.table("short_term_history")
        .select("content")
        .eq("user_id", user_id)
        .eq("role", "user")
        .gte("chat_time", since_iso)
        .order("chat_time", desc=True)
        .limit(limit)
        .execute()
    )
    rows = result.data or []
    return [str(r.get("content") or "") for r in rows if r.get("content")]


# ── Long-term Memory CRUD ────────────────────────────────────────────

def save_long_term_memory(
    user_id: str,
    memory_date: str,
    content: str,
    embedding: list[float],
    metadata: dict,
) -> dict:
    sb = get_supabase()
    row = {
        "user_id": user_id,
        "memory_date": memory_date,
        "content": content,
        "embedding": embedding,
        "metadata": metadata,
    }
    result = sb.table("long_term_memory").insert(row).execute()
    return result.data[0] if result.data else row


def list_recent_memories(user_id: str, limit: int = 300) -> list[dict]:
    """회상 후보 선별 등: 최근 장기기억을 memory_date 내림차순으로 가져온다."""
    sb = get_supabase()
    result = (
        sb.table("long_term_memory")
        .select("id, content, memory_date, metadata")
        .eq("user_id", user_id)
        .order("memory_date", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def get_happy_memories(
    user_id: str,
    *,
    min_importance: int = 6,
    limit: int = 5,
) -> list[dict]:
    """
    emotion이 행복/기쁨 계열이고 importance_score >= min_importance인 장기기억을 반환한다.
    중요도 내림차순으로 정렬하여 최상위 limit개를 돌려준다.
    """
    HAPPY_EMOTIONS = {"행복", "기쁨", "즐거움", "편안", "만족", "신남", "happy", "joy"}
    sb = get_supabase()
    result = (
        sb.table("long_term_memory")
        .select("id, content, memory_date, metadata")
        .eq("user_id", user_id)
        .order("memory_date", desc=False)
        .limit(500)
        .execute()
    )
    rows = result.data or []
    matched = [
        r for r in rows
        if int((r.get("metadata") or {}).get("importance_score") or 0) >= min_importance
        and (r.get("metadata") or {}).get("emotion", "") in HAPPY_EMOTIONS
    ]
    matched.sort(
        key=lambda r: int((r.get("metadata") or {}).get("importance_score") or 0),
        reverse=True,
    )
    return matched[:limit]


# ── Senior Profile CRUD ──────────────────────────────────────────────

def get_senior_profile(user_id: str) -> dict | None:
    sb = get_supabase()
    result = sb.table("senior_profile").select("*").eq("user_id", user_id).execute()
    return result.data[0] if result.data else None


def upsert_senior_profile(user_id: str, data: dict) -> dict:
    sb = get_supabase()
    row = {"user_id": user_id, **data}
    result = sb.table("senior_profile").upsert(row).execute()
    return result.data[0] if result.data else row


# ── Daily Care Report CRUD ───────────────────────────────────────────

def get_daily_report(user_id: str, report_date: str) -> dict | None:
    sb = get_supabase()
    result = (
        sb.table("daily_care_report")
        .select("*")
        .eq("user_id", user_id)
        .eq("report_date", report_date)
        .execute()
    )
    return result.data[0] if result.data else None


def upsert_daily_report(user_id: str, report_date: str, data: dict) -> dict:
    sb = get_supabase()
    existing = get_daily_report(user_id, report_date)
    if existing:
        result = (
            sb.table("daily_care_report")
            .update(data)
            .eq("report_id", existing["report_id"])
            .execute()
        )
    else:
        row = {"user_id": user_id, "report_date": report_date, **data}
        result = sb.table("daily_care_report").insert(row).execute()
    return result.data[0] if result.data else data


def get_weekly_reports(user_id: str, limit: int = 7) -> list[dict]:
    sb = get_supabase()
    result = (
        sb.table("daily_care_report")
        .select("*")
        .eq("user_id", user_id)
        .order("report_date", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def get_daily_reports_since(user_id: str, since_date: str, limit: int = 62) -> list[dict]:
    """report_date >= since_date 인 일간 리포트 (오름차순). 달력·구간 조회용."""
    sb = get_supabase()
    result = (
        sb.table("daily_care_report")
        .select("*")
        .eq("user_id", user_id)
        .gte("report_date", since_date)
        .order("report_date", desc=False)
        .limit(limit)
        .execute()
    )
    return result.data or []


# ── Intervention Action Log CRUD ─────────────────────────────────────

def log_intervention(
    user_id: str,
    action_type: str,
    trigger_emotion: str | None = None,
    suggested_content: str | None = None,
) -> dict:
    sb = get_supabase()
    row = {
        "user_id": user_id,
        "action_type": action_type,
        "trigger_emotion": trigger_emotion,
        "suggested_content": suggested_content,
    }
    result = sb.table("intervention_action_log").insert(row).execute()
    return result.data[0] if result.data else row


def update_intervention_response(action_id: int, user_response: str) -> dict:
    sb = get_supabase()
    result = (
        sb.table("intervention_action_log")
        .update({"user_response": user_response})
        .eq("action_id", action_id)
        .execute()
    )
    return result.data[0] if result.data else {}


def get_recent_interventions(user_id: str, limit: int = 10) -> list[dict]:
    sb = get_supabase()
    result = (
        sb.table("intervention_action_log")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


# ── Hybrid Search (Vector + GIN Metadata via RPC) ────────────────────

@dataclass
class HybridSearchResult:
    id: str
    user_id: str
    memory_date: str | None
    content: str
    metadata: dict
    vector_score: float
    metadata_score: float
    recency_score: float
    importance_score: float
    final_score: float


def hybrid_search(
    user_id: str,
    query_embedding: list[float],
    *,
    category: str | None = None,
    emotion: str | None = None,
    entities: list[str] | None = None,
    top_k: int = 15,
    vector_weight: float = 0.65,
    metadata_weight: float = 0.10,
    recency_weight: float = 0.15,
    importance_weight: float = 0.10,
) -> list[HybridSearchResult]:
    """Supabase RPC로 하이브리드 검색을 실행한다. ``long_term_memory``는 ``user_id``로만 필터된다."""
    sb = get_supabase()

    params: dict[str, Any] = {
        "p_user_id": user_id,
        "p_query_embedding": query_embedding,
        "p_top_k": top_k,
        "p_vector_weight": vector_weight,
        "p_metadata_weight": metadata_weight,
        "p_recency_weight": recency_weight,
        "p_importance_weight": importance_weight,
    }
    if category:
        params["p_category"] = category
    if emotion:
        params["p_emotion"] = emotion
    if entities:
        params["p_entities"] = entities

    result = sb.rpc("hybrid_search", params).execute()

    return [
        HybridSearchResult(
            id=r["id"],
            user_id=r["user_id"],
            memory_date=r.get("memory_date"),
            content=r["content"],
            metadata=r.get("metadata", {}),
            vector_score=float(r.get("vector_score", 0)),
            metadata_score=float(r.get("metadata_score", 0)),
            recency_score=float(r.get("recency_score", 0)),
            importance_score=float(r.get("importance_score", 0)),
            final_score=float(r.get("final_score", 0)),
        )
        for r in (result.data or [])
    ]


def check_duplicate_memory(
    user_id: str,
    content: str,
    query_embedding: list[float] | None = None,
    threshold: float = 0.80,
) -> bool:
    """새 기억이 기존 기억과 텍스트 수준에서 중복인지 확인한다.

    text-embedding-004 한국어 임베딩은 짧은 문장끼리 코사인 유사도가
    0.91~0.93 밴드에 몰리기 때문에 벡터 기반 중복 탐지가 부정확하다.
    텍스트 SequenceMatcher ratio >= threshold(0.80)이면 중복으로 판정.
    """
    from difflib import SequenceMatcher

    sb = get_supabase()
    result = (
        sb.table("long_term_memory")
        .select("id, content")
        .eq("user_id", user_id)
        .execute()
    )

    for r in (result.data or []):
        existing = r.get("content", "")
        ratio = SequenceMatcher(None, content, existing).ratio()
        if ratio >= threshold:
            logger.info(
                "중복 기억 차단 (text_ratio=%.3f): 기존='%s' vs 신규='%s'",
                ratio, existing[:60], content[:60],
            )
            return True
    return False
