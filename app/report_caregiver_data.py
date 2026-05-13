"""보호자 리포트: 마음 날씨 달력(14일)·월간 키워드 가공."""

from __future__ import annotations

import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import Any

KST = timezone(timedelta(hours=9))

def normalize_report_date_key(value: Any) -> str | None:
    """
    daily_care_report.report_date가 date·timestamp 문자열 등으로 올 때 YYYY-MM-DD로 통일.
    (예: '2026-04-10T00:00:00', '2026-04-10T00:00:00+00:00' → '2026-04-10')
    """
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        try:
            iv = value.isoformat()
            if isinstance(iv, str) and len(iv) >= 10:
                return iv[:10]
        except (TypeError, ValueError):
            pass
    s = str(value).strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return None


_KR_STOP = frozenset(
    """
    은 는 이 가 을 를 에 의 와 과 도 만 보 다 때 거 등 좀 잘 게 음 응 그 이거 그거 저거 뭐
    예요 아요 해요 네요 습니다 요 죠 지 만요 은요 는요 이요 가요 을요 를요
    안녕 하세요 감사합니다 그리고 그런데 그래서 정말 너무 매우 아주 조금 오늘 같아 내가 때문에 기분이 있는
    """.split()
)


def build_mood_weather_calendar_14d(
    today: date,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    최근 14일(오늘 포함) 마음 날씨 달력.
    rows: [{report_date, mood_avg_score}, ...]
    score: DB 원점수 mood_avg_score(-30~100). 0~100 환산·색 구간은 프론트에서 처리.
    """
    score_by_date: dict[str, float | None] = {}
    for r in rows:
        ds = normalize_report_date_key(r.get("report_date"))
        if not ds:
            continue
        raw = r.get("mood_avg_score")
        if raw is None:
            score_by_date[ds] = None
        else:
            try:
                score_by_date[ds] = float(raw)
            except (TypeError, ValueError):
                score_by_date[ds] = None

    wd_ko = ["월", "화", "수", "목", "금", "토", "일"]
    days: list[dict[str, Any]] = []
    for i in range(13, -1, -1):
        dd = today - timedelta(days=i)
        ds = dd.isoformat()
        sc_val = score_by_date.get(ds)
        days.append(
            {
                "date": ds,
                "day": dd.day,
                "weekday": wd_ko[dd.weekday()],
                "score": round(sc_val, 1) if sc_val is not None else None,
            }
        )

    start_d = today - timedelta(days=13)
    return {
        "kind": "14d",
        "start_date": start_d.isoformat(),
        "end_date": today.isoformat(),
        "days": days,
    }


def extract_monthly_keywords(texts: list[str], top_n: int = 5) -> list[dict[str, Any]]:
    """간단 한국어 연속 음절 추출 + 빈도."""
    counter: Counter[str] = Counter()
    for t in texts:
        if not t:
            continue
        for w in re.findall(r"[가-힣]{2,}", t):
            if w in _KR_STOP or len(w) < 2:
                continue
            counter[w] += 1
    return [{"word": w, "count": n} for w, n in counter.most_common(top_n)]


def month_start_iso_for_chat_filter(year: int, month: int) -> str:
    """Supabase chat_time 비교용 (월초 KST)."""
    dt = datetime(year, month, 1, 0, 0, 0, tzinfo=KST)
    return dt.isoformat()
