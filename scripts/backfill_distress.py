#!/usr/bin/env python3
"""
user_001 distress 역산 backfill.

- short_term_history의 전체 user 발화를 날짜 순으로 시뮬레이션
- emotion_label 있으면 사용, 없으면 키워드 매칭
- 일자별 로직:
    - 하루가 넘어가면 distress_score *= 0.7 (경과일 수만큼)
    - distress_score_daily = 50 으로 리셋
    - 10분 침묵 = 세션 종료 → 세션 점수를 distress에 정산
- 각 일자 daily_care_report upsert
- 마지막 일자 기준 senior_profile 업데이트
- 앞으로: senior_profile.last_decay_date 가 정확해지면 기존 코드가 올바르게 동작함
"""
from __future__ import annotations

import logging
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import get_supabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

USER_ID = "user_001"
KST = ZoneInfo("Asia/Seoul")

# ── 상수 (emotion.py와 동일) ─────────────────────────────────────────
DECAY_RATE = 0.7
SESSION_GAP = timedelta(minutes=10)
DISTRESS_MIN, DISTRESS_MAX = -30.0, 100.0
SESSION_MIN, SESSION_MAX = -10.0, 10.0
DAILY_DEFAULT = 50.0
DAILY_MIN, DAILY_MAX = 0.0, 100.0

EMOTION_SCORES: dict[str, int] = {
    "공포": 6, "불안": 6,
    "슬픔": 4, "우울": 4,
    "분노": 3,
    "혐오": 2,
    "놀람": 0,
    "중립": -1,
    "기쁨": -5, "행복": -5,
}

DANGER_EMOTIONS = {"공포", "슬픔", "우울", "불안"}

CARE_THRESHOLDS = [(65, "DANGER"), (35, "WARNING")]


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _care_level(score: float) -> str:
    for threshold, level in CARE_THRESHOLDS:
        if score >= threshold:
            return level
    return "NORMAL"


def _mood_state(daily: float) -> str:
    if daily >= 75:
        return "우울"
    if daily >= 60:
        return "적적함"
    if daily >= 40:
        return "보통"
    if daily >= 25:
        return "편안함"
    return "기분좋음"


# ── 키워드 기반 감정 감지 ────────────────────────────────────────────
_KW_MAP: list[tuple[str, list[str]]] = [
    ("공포",  ["무서", "두려", "공포", "조마"]),
    ("불안",  ["불안", "걱정", "초조", "두렵"]),
    ("슬픔",  ["슬프", "슬픔", "눈물", "서럽", "서운", "외로", "상실", "허전", "우울", "아쉽",
               "서글", "애달", "속상", "찡", "먹먹", "쓸쓸"]),
    ("분노",  ["화가", "화나", "짜증", "분하", "억울", "어이없", "미워", "터진", "구역질",
               "치가", "분개", "허탈", "절망", "답답", "못마", "열받", "성가"]),
    ("혐오",  ["혐오", "역겨"]),
    ("기쁨",  ["기쁘", "기뻐", "기쁨", "신나", "신이", "행복", "편안", "다행", "감사", "고마",
               "즐거", "뿌듯", "반가", "좋아", "흐뭇", "뿌듯"]),
    ("놀람",  ["놀랐", "깜짝", "놀라"]),
]


def detect_emotion(content: str, emotion_label: str | None) -> str:
    if emotion_label and emotion_label.strip() in EMOTION_SCORES:
        return emotion_label.strip()
    text = re.sub(r"^HS\d+:\s*", "", content).strip()
    for emotion, keywords in _KW_MAP:
        if any(kw in text for kw in keywords):
            return emotion
    return "중립"


# ── DB 접근 ──────────────────────────────────────────────────────────

def fetch_user_messages() -> list[dict]:
    sb = get_supabase()
    rows = (
        sb.table("short_term_history")
        .select("chat_id, chat_time, content, emotion_label")
        .eq("user_id", USER_ID)
        .eq("role", "user")
        .order("chat_time", desc=False)
        .execute()
    )
    return rows.data or []


def upsert_daily_report(day: date, data: dict) -> None:
    sb = get_supabase()
    day_str = day.isoformat()
    existing = (
        sb.table("daily_care_report")
        .select("report_id")
        .eq("user_id", USER_ID)
        .eq("report_date", day_str)
        .execute()
    )
    if existing.data:
        sb.table("daily_care_report").update(data).eq("report_id", existing.data[0]["report_id"]).execute()
    else:
        sb.table("daily_care_report").insert({"user_id": USER_ID, "report_date": day_str, **data}).execute()


def update_senior_profile(data: dict) -> None:
    sb = get_supabase()
    sb.table("senior_profile").update(data).eq("user_id", USER_ID).execute()


# ── 메인 시뮬레이션 ──────────────────────────────────────────────────

def main() -> None:
    messages = fetch_user_messages()
    if not messages:
        logger.error("메시지 없음")
        return

    logger.info("총 %d 건 발화 로드", len(messages))

    # chat_time → KST datetime 변환
    def to_kst(ts: str) -> datetime:
        # Python 3.10 fromisoformat는 +00:00 형식만 지원 → 6자리 마이크로초 포함 포맷 처리
        s = ts.strip()
        # '+00:00'·'+09:00' 등 offset 포함 포맷 직접 파싱
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            # 예: '2026-03-28T06:04:47.35988+00:00' (5자리 마이크로초)
            # datetime.strptime 으로 재시도
            import re as _re
            # timezone offset 분리
            m = _re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)([+-]\d{2}:\d{2}|Z)?$", s)
            if m:
                base, tz_str = m.group(1), m.group(2) or "+00:00"
                # 마이크로초를 6자리로 맞춤
                if "." in base:
                    main_part, frac = base.split(".", 1)
                    base = f"{main_part}.{frac.ljust(6, '0')[:6]}"
                offset_h, offset_m = int(tz_str[1:3]), int(tz_str[4:6])
                sign = 1 if tz_str[0] == "+" else -1
                offset = timezone(sign * timedelta(hours=offset_h, minutes=offset_m))
                dt = datetime.fromisoformat(base).replace(tzinfo=offset)
            else:
                raise
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(KST)

    def to_kst_date(ts: str) -> date:
        return to_kst(ts).date()

    # 일자별 그룹
    by_day: dict[date, list[dict]] = defaultdict(list)
    for m in messages:
        by_day[to_kst_date(m["chat_time"])].append(m)

    sorted_days = sorted(by_day)

    # 초기 상태
    distress = 0.0
    distress_daily = DAILY_DEFAULT
    session_score = 0.0
    last_decay_date: date = sorted_days[0] - timedelta(days=1)
    last_msg_at: datetime | None = None

    for day in sorted_days:
        day_messages = by_day[day]

        # ── 일간 감쇄: 전날 누적 distress × 0.7^경과일 ───────────
        days_gap = (day - last_decay_date).days
        if days_gap >= 1:
            if last_msg_at is not None:
                # 전날 열려있던 세션 정산 (이미 10분 이상 경과했으므로)
                clamped = _clamp(session_score, SESSION_MIN, SESSION_MAX)
                distress = _clamp(distress + clamped, DISTRESS_MIN, DISTRESS_MAX)
                session_score = 0.0
            for _ in range(days_gap):
                distress *= DECAY_RATE
            distress_daily = DAILY_DEFAULT
            last_decay_date = day
            logger.info("[%s] 감쇄 적용(%d일) distress=%.2f, daily 리셋=50", day, days_gap, distress)

        # ── 해당 일자 메시지 처리 ────────────────────────────────
        day_utterances = 0
        day_danger_count = 0
        day_dominant_emotion = "중립"

        # 세션 기반 mood_avg: 세션 종료 시점의 daily 값만 샘플로 사용
        mood_session_avg = DAILY_DEFAULT
        mood_session_count = 0

        for msg in day_messages:
            msg_at = to_kst(msg["chat_time"])
            emotion = detect_emotion(msg["content"], msg.get("emotion_label"))
            delta = EMOTION_SCORES.get(emotion, 0)

            # 세션 경계 판정 → 이전 세션의 daily 스냅샷을 mood 샘플로 기록
            if last_msg_at is not None and (msg_at - last_msg_at) >= SESSION_GAP:
                # 세션 종료: 현재 daily를 샘플로 기록 (발화 처리 전)
                mood_session_count += 1
                mood_session_avg += (distress_daily - mood_session_avg) / mood_session_count

                clamped = _clamp(session_score, SESSION_MIN, SESSION_MAX)
                distress = _clamp(distress + clamped, DISTRESS_MIN, DISTRESS_MAX)
                session_score = 0.0
                logger.debug("  세션 종료 정산: distress=%.2f, session_daily=%.1f", distress, distress_daily)

            session_score += delta
            distress_daily = _clamp(distress_daily + delta, DAILY_MIN, DAILY_MAX)

            last_msg_at = msg_at
            day_utterances += 1
            if emotion in DANGER_EMOTIONS:
                day_danger_count += 1
            day_dominant_emotion = emotion

        # 해당 일자 마지막 세션 (아직 닫히지 않은 열린 세션)도 샘플로 기록
        mood_session_count += 1
        mood_session_avg += (distress_daily - mood_session_avg) / mood_session_count

        # effective distress = 현재 distress + 현재 세션 (클램프)
        effective = _clamp(
            distress + _clamp(session_score, SESSION_MIN, SESSION_MAX),
            DISTRESS_MIN, DISTRESS_MAX,
        )
        care = _care_level(effective)
        requires_check = care == "DANGER" or day_danger_count >= 3

        upsert_daily_report(day, {
            "dominant_emotion": day_dominant_emotion,
            "mood_avg_score": round(mood_session_avg, 2),
            "today_care_level": care,
            "total_utterance": day_utterances,
            "danger_count": day_danger_count,
            "requires_check": requires_check,
            "session_count": mood_session_count,
        })
        logger.info(
            "[%s] utterances=%d, sessions=%d, danger=%d, daily=%.1f, mood_avg=%.1f, distress=%.2f, session=%.1f, care=%s",
            day, day_utterances, mood_session_count, day_danger_count, distress_daily,
            mood_session_avg, distress, session_score, care,
        )

    # ── 마지막 상태: senior_profile 업데이트 ─────────────────────────
    # 세션은 아직 닫지 않은 상태 → 이후 실제 메시지가 오면 코드가 자동 처리
    # current_session_score 는 클램프 범위 내 값으로만 저장
    final_session = _clamp(session_score, SESSION_MIN, SESSION_MAX)
    final_effective = _clamp(distress + final_session, DISTRESS_MIN, DISTRESS_MAX)
    final_care = _care_level(final_effective)
    final_mood = _mood_state(distress_daily)

    profile_data = {
        "distress_score": round(distress, 2),
        "distress_score_daily": round(distress_daily, 2),
        "care_level": final_care,
        "current_session_score": round(final_session, 2),
        "last_decay_date": last_decay_date.isoformat(),
        "mood_state": final_mood,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    update_senior_profile(profile_data)
    logger.info("senior_profile 업데이트 완료: %s", profile_data)


if __name__ == "__main__":
    main()
