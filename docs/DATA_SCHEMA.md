# 데이터 스키마 (코드 기준 역추적)

본 문서는 애플리케이션이 Supabase(PostgreSQL)에 가정하는 **테이블·컬럼·RPC**를 소스 코드에서 역추적한 것이다. **실제 DDL·제약·인덱스**는 Supabase 프로젝트에 있으며, 마이그레이션 파일이 본 저장소에 없을 수 있으니 배포 시 콘솔·`pg_dump`와 대조한다.

---

## 1. 아키텍처 요약

| 구분 | 기술 |
|------|------|
| 앱 → DB (대부분) | **Supabase PostgREST** (`supabase-py` `Client`) |
| 벡터 검색 | 테이블 `long_term_memory` + **RPC** `hybrid_search` (pgvector·GIN 등은 DB 측 정의) |
| 직접 SQL 연결 문자열 | `SUPABASE_DB_URL` — 앱 런타임 CRUD에는 미사용(주로 운영·마이그레이션) |

---

## 2. 테이블 정의 (논리)

### 2.1 `short_term_history`

단기 대화 로그. 장기기억 추출 파이프라인의 입력.

| 컬럼 | 타입(추정) | 필수 | 설명 |
|------|-------------|------|------|
| `chat_id` | `uuid` (PK) | ✓ | insert 시 DB 기본값 또는 자동 생성으로 반환된다고 가정 |
| `user_id` | `text` | ✓ | 사용자 식별자 |
| `role` | `text` | ✓ | `"user"` \| `"assistant"` |
| `content` | `text` | ✓ | 메시지 본문 |
| `chat_time` | `timestamptz` | ✓ | 정렬·구간 조회에 사용(기본 `now()` 추정) |
| `is_extracted` | `boolean` | ✓ | 기본 `false`, 추출 완료 시 `true` |
| `emotion_label` | `text` | | **선택**. 백필 스크립트 등에서 조회(`scripts/backfill_distress.py`). 런타임 insert에는 미포함 |

**인덱스(권장, 코드에서 암시)**  
- `(user_id, is_extracted, chat_time)`  
- `(user_id, role, chat_time DESC)`

---

### 2.2 `long_term_memory`

장기기억 + 벡터. RAG·회상·아침 브리프에서 사용.

| 컬럼 | 타입(추정) | 필수 | 설명 |
|------|-------------|------|------|
| `id` | `uuid` | ✓ | PK |
| `user_id` | `text` | ✓ | 사용자별 격리 |
| `memory_date` | `date` 또는 `text` | ✓ | `YYYY-MM-DD` 문자열로 저장·정렬 |
| `content` | `text` | ✓ | 3인칭 요약 등 |
| `embedding` | `vector` | ✓ | `text-embedding-004` 차원에 맞춤(pgvector) |
| `metadata` | `jsonb` | ✓ | 아래 **메타 스키마** 참고 |

#### `metadata` JSON (앱이 쓰는 키)

```json
{
  "category": "Family | Health | Routine | Preference | Event | Emotion",
  "emotion": "string",
  "importance_score": 1,
  "entity": ["string"]
}
```

---

### 2.3 `senior_profile`

사용자당 1행(upsert 키: `user_id`). 정서 엔진·개입 피로도.

| 컬럼 | 타입(추정) | 설명 |
|------|-------------|------|
| `user_id` | `text` (PK) | |
| `distress_score` | `float` | 누적 정서 피로도, 대략 `-30` ~ `100` |
| `distress_score_daily` | `float` | 일간 기분 `0` ~ `100`, 기본 `50` |
| `care_level` | `text` | `NORMAL` \| `WARNING` \| `DANGER` |
| `current_session_score` | `float` | 세션 내 누적 감정 가중 |
| `last_message_at` | `timestamptz` | 마지막 발화 시각(세션 경계) |
| `last_decay_date` | `date` | 일간 감쇠 기준일 |
| `mood_state` | `text` | 예: `우울`, `적적함`, `보통`, `편안함`, `기분좋음` |
| `recent_action_fatigue` | `float` | `0` ~ `1` |
| `updated_at` | `timestamptz` | |

**선택·기타(코드에서 참조 가능)**  

- `name` — `GET /welcome` 등에서 DB 이름으로 사용 가능  
- 개입 피드백 후 `last_accepted_action` / `last_rejected_action` 갱신 시도(`emotion.py` upsert payload). DB 컬럼 존재 여부는 Supabase 스키마와 일치해야 함.

---

### 2.4 `daily_care_report`

일자별 케어 요약. `report_id`로 업데이트.

| 컬럼 | 타입(추정) | 설명 |
|------|-------------|------|
| `report_id` | `uuid` 또는 `bigint` (PK) | insert/update 시 사용 |
| `user_id` | `text` | |
| `report_date` | `date` | `YYYY-MM-DD` |
| `mood_avg_score` | `float` | 세션 종료 시 이동평균 반영 |
| `session_count` | `int` | 세션 샘플 수 |
| `dominant_emotion` | `text` | 최근 발화 감정 |
| `today_care_level` | `text` | 당일 케어 등급 |
| `total_utterance` | `int` | 발화 수 |
| `danger_count` | `int` | 위험 감정 라벨 누적 |
| `requires_check` | `boolean` | 보호자 점검 필요 플래그 |

---

### 2.5 `intervention_action_log`

행동 개입 이력.

| 컬럼 | 타입(추정) | 설명 |
|------|-------------|------|
| `action_id` | `bigint` 또는 `serial` (PK) | 피드백 API에서 `int`로 참조 |
| `user_id` | `text` | |
| `action_type` | `text` | 예: `VOD_추천`, `음악_재생` |
| `trigger_emotion` | `text` | nullable |
| `suggested_content` | `text` | nullable |
| `user_response` | `text` | nullable, `ACCEPT` / `REJECT` / `IGNORE` 등 |
| `created_at` | `timestamptz` | 정렬 기준 |

---

## 3. RPC: `hybrid_search`

`app/database.py`의 `hybrid_search()` 호출 파라미터. **DB에 동일 시그니처의 함수**가 있어야 한다.

| 파라미터 | 타입(추정) | 필수 | 설명 |
|----------|------------|------|------|
| `p_user_id` | `text` | ✓ | |
| `p_query_embedding` | `vector` / `float8[]` | ✓ | 쿼리 임베딩 |
| `p_top_k` | `int` | ✓ | |
| `p_vector_weight` | `float` | ✓ | |
| `p_metadata_weight` | `float` | ✓ | |
| `p_recency_weight` | `float` | ✓ | |
| `p_importance_weight` | `float` | ✓ | |
| `p_category` | `text` | | 선택 필터 |
| `p_emotion` | `text` | | 선택 필터 |
| `p_entities` | `text[]` | | 선택 필터 |

**반환 행(코드에서 읽는 키)**  
`id`, `user_id`, `memory_date`, `content`, `metadata`, `vector_score`, `metadata_score`, `recency_score`, `importance_score`, `final_score`

---

## 4. 엔티티 관계(개략)

```
user_id (논리 키)
    ├── short_term_history (1:N)
    ├── long_term_memory (1:N)
    ├── senior_profile (1:1)
    ├── daily_care_report (1:N, by report_date)
    └── intervention_action_log (1:N)
```

---

## 5. 문서 이력

| 일자 | 내용 |
|------|------|
| 2026-05-13 | 코드 기준 초안 작성 |
