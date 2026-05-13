# 요구사항 명세서 (코드 기준 역추적)

본 문서는 저장소 **현행 구현**(`app/`, `static/`)을 기준으로 기능·비기능 요구사항을 정리한 것이다. 별도 기획서와 불일치할 수 있으며, 변경 시 코드와 함께 본 문서를 갱신한다.

---

## 1. 시스템 개요

| 항목 | 내용 |
|------|------|
| 시스템 명 | 마음 — 시니어 감정 맞춤형 RAG 말벗 (백엔드 + 정적 웹 UI) |
| 대상 사용자 | 60대 이상 독거 어르신(대화 상대), 보호자(리포트 뷰) |
| 주요 가치 | 공감형 대화, 장기기억 기반 RAG, 정서 피로도·일간 리포트, TV/음성 사용 시나리오 |

---

## 2. 외부 연동·인프라

| ID | 요구사항 | 구현 참조 |
|----|----------|-----------|
| NFR-EXT-01 | Supabase **PostgREST**로 테이블 CRUD, **RPC** `hybrid_search`로 벡터+메타 하이브리드 검색 | `app/database.py` |
| NFR-EXT-02 | Google **Vertex AI 경유** Gemini(`gemini-2.5-flash`) — 챗, 기억 추출, 일정 파싱, 회상 문구 등 | `app/rag.py`, `app/memory_agent.py`, `app/main.py` 등 |
| NFR-EXT-03 | Google **Cloud Text-to-Speech** — 한국어 MP3 합성 | `app/main.py` `/tts` |
| NFR-EXT-04 | Google **Cloud Speech-to-Text** — 업로드 오디오 텍스트화 | `app/stt.py`, `/stt` |
| NFR-EXT-05 | 기상청 허브 API — 서울 기온·습도·강수 | `app/main.py` `/weather` |
| NFR-ENV-01 | `GCP_PROJECT_ID`, `GCP_LOCATION`, `GOOGLE_APPLICATION_CREDENTIALS`, `SUPABASE_DB_URL`, `SUPABASE_URL`, `SUPABASE_KEY` | `app/config.py`, `.env` |

---

## 3. 기능 요구사항

### 3.1 대화·RAG

| ID | 요구사항 | 상세 | API/모듈 |
|----|----------|------|----------|
| FR-CHAT-01 | 사용자 발화를 **단기 이력**에 저장한다 | `user` 역할, `user_id` 단위 | `POST /chat` 내부 |
| FR-CHAT-02 | **하이브리드 검색**으로 장기기억을 조회한 뒤 공감형 답변을 생성한다 | 임베딩 쿼리 + RPC 가중치(벡터·메타·최신·중요도) | `app/rag.py`, `hybrid_search` |
| FR-CHAT-03 | 답변에 **안전 플래그**(예: DEPRESSION, ANGER)와 감정·케어등급·정서 점수를 포함한다 | LLM JSON 출력 파싱 | `ChatResponse` |
| FR-CHAT-04 | 봇 답변을 단기 이력에 **비동기 저장**한다 | 응답 후 백그라운드 | `_save_assistant_message_bg` |
| FR-CHAT-05 | 응답 후 **장기기억 추출**을 백그라운드로 시도한다(락으로 중복 완화) | 미추출 대화 청크 → LLM → 벡터 저장 | `_background_extract`, `app/memory_agent.py` |
| FR-CHAT-06 | 요청에 **프로필 힌트**(이름, 성별, 나이, 아바타)를 선택적으로 넘길 수 있다 | RAG 프롬프트 맥락 | `ChatRequest` |

### 3.2 정서·일간 리포트

| ID | 요구사항 | 상세 | 모듈 |
|----|----------|------|------|
| FR-EMO-01 | 발화별 감정에 따라 **누적 정서피로도**·**일간 기분**·**세션 점수**를 갱신한다 | 10분 침묵 세션 종료, 일자 경과 시 감쇠 등 | `app/emotion.py` |
| FR-EMO-02 | **케어 등급** NORMAL / WARNING / DANGER 를 산출한다 | effective distress 임계(35, 65) | `emotion.py` |
| FR-EMO-03 | **일간 케어 리포트**에 발화 수, 위험 감정 횟수, 당일 케어등급, 세션 기반 mood 평균 등을 반영한다 | `daily_care_report` | `emotion.py` |

### 3.3 기억 추출

| ID | 요구사항 | 상세 | API |
|----|----------|------|-----|
| FR-MEM-01 | 미추출 단기 대화를 LLM으로 분석해 **장기기억** 후보를 만든다 | 카테고리·감정·중요도·엔티티 | `memory_agent.py` |
| FR-MEM-02 | 기억은 **임베딩**과 **JSON 메타데이터**와 함께 저장된다 | 중복은 텍스트 유사도로 차단 | `save_long_term_memory`, `check_duplicate_memory` |
| FR-MEM-03 | 처리된 단기 메시지는 **추출 완료**로 표시한다 | `is_extracted` | `mark_as_extracted` |
| FR-MEM-04 | 수동/배치로 기억 추출 API를 호출할 수 있다 | `user_id` 선택, `limit` | `POST /extract-memory` |

### 3.4 음성·미디어

| ID | 요구사항 | 상세 | API |
|----|----------|------|-----|
| FR-AUD-01 | 텍스트를 **한국어 TTS(MP3)** 로 반환한다 | 이모지·마크다운 등 제거 후 합성 | `POST /tts` |
| FR-AUD-02 | 업로드 오디오(WebM/Opus, WAV 등)를 **STT** 로 변환한다 | `ko-KR`, `latest_long` | `POST /stt` |

### 3.5 프로필·브리프·인사

| ID | 요구사항 | 상세 | API |
|----|----------|------|-----|
| FR-PRO-01 | `user_id` 로 **시니어 프로필**을 조회한다 | 없으면 404 | `GET /profile/{user_id}` |
| FR-PRO-02 | **아침 브리프**: effective distress 구간별 회상 모드·행복 기억 목록·보호자 알림 플래그 | DANGER 시 `requires_check` 갱신 | `GET /morning-brief/{user_id}` |
| FR-PRO-03 | **웰컴 메시지**: 정서 점수가 임계 이상이면 긍정 회상 기반 LLM 인사 | 그 외 기본 문구 | `GET /welcome/{user_id}` |

### 3.6 일정·자연어

| ID | 요구사항 | 상세 | API |
|----|----------|------|-----|
| FR-SCH-01 | 사용자 문장에서 **일정 추가 의도**와 시간·설명·반복 여부를 추출한다 | JSON 단일 출력 | `POST /parse-schedule` |

### 3.7 리포트 UI용 데이터

| ID | 요구사항 | 상세 | API |
|----|----------|------|-----|
| FR-RPT-01 | 최근 N일 **일간 리포트** 목록을 반환한다 | 기본 7일 | `GET /reports/{user_id}` |
| FR-RPT-02 | 리포트 화면용 **가공 데이터**(주간 점수, 오늘 vs 평균, 회상, 14일 달력, 월간 키워드)를 한 번에 반환한다 | LLM 키워드 실패 시 휴리스틱 | `GET /report-summary/{user_id}` |

### 3.8 행동 개입

| ID | 요구사항 | 상세 | API |
|----|----------|------|-----|
| FR-INT-01 | 개입 행위를 **로그에 남긴다** | action_type, trigger, 제안 내용 | `POST /intervention` |
| FR-INT-02 | 사용자 반응(ACCEPT/REJECT/IGNORE)을 기록하고 **추천 피로도**를 갱신한다 | 프로필 `recent_action_fatigue` | `POST /intervention/feedback` |
| FR-INT-03 | 최근 개입 이력을 조회한다 | | `GET /interventions/{user_id}` |

### 3.9 기타·운영

| ID | 요구사항 | 상세 | API |
|----|----------|------|-----|
| FR-OPS-01 | 헬스 체크 | | `GET /health` |
| FR-OPS-02 | 루트에서 **메인 SPA HTML** 제공, `/static` 정적 자원 마운트 | HTML 캐시 완화 헤더 | `app/main.py` |
| FR-OPS-03 | CORS 전체 허용(개발 편의) | `allow_origins=["*"]` | `main.py` |

### 3.10 웹 클라이언트(TV 시나리오, `static/index.html`)

| ID | 요구사항 | 상세 |
|----|----------|------|
| FR-UI-01 | **웨이크워드** 「마음아」 인식 후 대화 전송(Web Speech API, continuous) | 우측 패널 텍스트 입력·마이크 버튼 제거, 힌트 바 표시 |
| FR-UI-02 | 챗봇 응답 **TTS** 재생, 토글로 끄기 | `POST /tts`, 실패 시 `speechSynthesis` |
| FR-UI-03 | 온보딩(이름·성별·나이·아바타), 설정, **정서 리포트** 오버레이 | `GET /report-summary`, `/profile` 등 |
| FR-UI-04 | 일정은 **로컬스토리지** 위주(백엔드 일정 테이블과 별개) | 채팅에서 일정 문구 감지 시 `/parse-schedule` 연동 로직 존재 |

---

## 4. 비기능·제약

| ID | 내용 |
|----|------|
| NFR-REL-01 | 장기기억 추출·봇 메시지 저장 실패는 로그만 남기고 사용자 응답은 가능한 한 성공 경로 유지 |
| NFR-REL-02 | RAG/저장 실패 시 `/chat` 은 502 |
| NFR-SEC-01 | API 키·DB URL은 환경변수 관리, 저장소에 실비밀 커밋 금지 |
| NFR-LIM-01 | 메인 화면 STT는 **브라우저 Web Speech API**에 의존(Google Cloud STT 아님) |
| NFR-LIM-02 | `senior_profile` 의 표시용 이름 등은 **프론트 로컬 프로필**과 DB가 완전 동기화되지 않을 수 있음(`GET /profile`은 DB만) |

---

## 5. API 요약

| Method | Path | 설명 |
|--------|------|------|
| POST | `/chat` | RAG 대화 |
| POST | `/extract-memory` | 장기기억 추출 배치 |
| POST | `/tts` | TTS |
| POST | `/stt` | STT |
| GET | `/profile/{user_id}` | 프로필 |
| GET | `/morning-brief/{user_id}` | 아침 브리프 |
| GET | `/welcome/{user_id}` | 인사 문구 |
| POST | `/parse-schedule` | 일정 의도 파싱 |
| GET | `/reports/{user_id}` | 일간 리포트 목록 |
| GET | `/report-summary/{user_id}` | 리포트 UI 묶음 데이터 |
| POST | `/intervention` | 개입 기록 |
| POST | `/intervention/feedback` | 개입 피드백 |
| GET | `/interventions/{user_id}` | 개입 목록 |
| GET | `/weather` | 서울 날씨 |
| GET | `/health` | 헬스 |
| GET | `/` | `index.html` |

---

## 6. 문서 이력

| 일자 | 내용 |
|------|------|
| 2026-05-13 | 코드 기준 초안 작성 |
