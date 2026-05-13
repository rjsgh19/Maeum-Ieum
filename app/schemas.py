"""
Pydantic 모델: API Request/Response + LLM 기억 추출 결과 파싱.
"""

from datetime import date, datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── Chat API ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_id: str
    message: str
    category: Optional[str] = Field(
        None,
        description="검색 필터용 카테고리 (Family, Health, Routine, Preference, Event, Emotion)",
    )
    emotion: Optional[str] = Field(
        None,
        description="검색 필터용 감정 (기쁨, 슬픔, 외로움, 아쉬움 등)",
    )
    display_name: Optional[str] = Field(
        None,
        description="사용자 이름(호칭용)",
    )
    gender: Optional[str] = Field(
        None,
        description="성별: 남성, 여성, 기타",
    )
    age: Optional[int] = Field(
        None,
        ge=1,
        le=120,
        description="나이(세)",
    )
    avatar_id: Optional[str] = Field(
        None,
        description="말벗 캐릭터: boy, girl, dog, cat, robot",
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="어르신께 전달할 답변 텍스트")
    alert: Optional[str] = Field(
        None,
        description="안전 플래그 (DEPRESSION, ANGER 등). 없으면 null",
    )
    detected_emotion: Optional[str] = Field(
        None,
        description="LLM이 감지한 사용자 감정 (기쁨, 슬픔, 분노, 공포, 혐오, 놀람, 중립)",
    )
    care_level: Optional[str] = Field(
        None,
        description="케어 등급 (NORMAL, WARNING, DANGER)",
    )
    distress_score: Optional[float] = Field(
        None,
        description="누적 정서피로도 (-30 ~ 100)",
    )
    distress_score_daily: Optional[float] = Field(
        None,
        description="일간 기분 점수 (0 ~ 100)",
    )
    retrieved_memories: list[str] = Field(
        default_factory=list,
        description="RAG로 참조한 과거 기억 목록",
    )


# ── Memory Extraction ────────────────────────────────────────────────

class ExtractedMemoryItem(BaseModel):
    memory_date: str = Field(..., description="YYYY-MM-DD 형식 날짜")
    content: str = Field(..., description="3인칭 관찰자 시점 요약")
    category: str = Field(..., description="Family|Health|Routine|Preference|Event|Emotion")
    emotion: str = Field(..., description="주된 감정")
    importance_score: int = Field(..., ge=1, le=10)
    entity: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    memory: list[ExtractedMemoryItem]


class ExtractMemoryRequest(BaseModel):
    user_id: Optional[str] = Field(
        None,
        description="특정 user_id만 추출. 비우면 전체 미추출 대화 대상",
    )
    limit: int = Field(50, ge=1, le=200)


class ExtractMemoryResponse(BaseModel):
    extracted_count: int
    user_id: Optional[str] = None
    memories: list[ExtractedMemoryItem] = Field(default_factory=list)


# ── Schedule Parse ───────────────────────────────────────────────────

class ParseScheduleRequest(BaseModel):
    text: str = Field(..., description="사용자 발화 원문")


class ParseScheduleResponse(BaseModel):
    is_schedule: bool
    time: Optional[str] = Field(None, description="HH:MM 24시간제. 시간 없으면 null")
    desc: Optional[str] = Field(None, description="간결한 일정 설명")
    recurring: bool = Field(False, description="매일/반복 루틴이면 true, 당일 일정이면 false")


# ── Shared / Misc ────────────────────────────────────────────────────

class ChatMessageOut(BaseModel):
    chat_id: UUID
    user_id: str
    role: str
    content: str
    chat_time: datetime

    model_config = {"from_attributes": True}
