# DX_Team4 — 마음 (시니어 감정 맞춤형 RAG 챗봇)

60대 이상 독거 어르신을 위한 **음성·텍스트 대화**, **장기기억(RAG)**, **보고서·추억 생성** 등을 제공하는 **FastAPI** 백엔드와 **정적 웹 UI**(`static/`)로 구성된 프로젝트입니다.

## 사전 준비

- **Python** 3.11 이상 권장 (3.12·3.13 호환 확인됨)
- **Google Cloud**: Vertex AI(Gemini), Cloud Speech-to-Text, Cloud Text-to-Speech 사용을 전제로 합니다. 프로젝트 ID·리전 설정과 **서비스 계정 키**(`GOOGLE_APPLICATION_CREDENTIALS`)가 필요합니다.
- **Supabase**: PostgreSQL 연결 문자열(`SUPABASE_DB_URL`) 및 PostgREST용 `SUPABASE_URL`, `SUPABASE_KEY`가 필요합니다. (값은 팀/개인 인스턴스 기준으로 `.env`에 직접 넣습니다.)

## 설치

저장소 루트(`DX_Team4-main/`)에서 가상환경을 만들고 의존성을 설치합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 환경 변수

1. `.env.example`을 복사해 `.env`를 만듭니다.  
   `cp .env.example .env`
2. `.env` 안의 값을 **본인 GCP·Supabase** 정보로 바꿉니다. (예시 파일에 들어 있는 URL·키는 그대로 쓰지 말고, 반드시 교체하세요.)

애플리케이션에서 읽는 주요 변수는 다음과 같습니다. (`app/config.py`의 `Settings`와 동일한 이름입니다.)

| 변수 | 설명 |
|------|------|
| `GCP_PROJECT_ID` | GCP 프로젝트 ID |
| `GCP_LOCATION` | Vertex 리전 (기본 예: `asia-northeast3`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | 서비스 계정 JSON 키 파일 경로 |
| `SUPABASE_DB_URL` | Supabase PostgreSQL 접속 URL (pgvector·RPC 등 DB 기능용) |
| `SUPABASE_URL` | Supabase 프로젝트 URL (REST 클라이언트용) |
| `SUPABASE_KEY` | Supabase anon(또는 서비스) 키 |

GCP 콘솔에서 **Cloud Speech-to-Text API** 등 사용 API를 켜고, 서비스 계정에 필요한 IAM 권한을 부여해야 STT·TTS·LLM 호출이 동작합니다.

## 실행

**반드시 프로젝트 루트**에서 모듈 경로 `app`이 보이도록 실행합니다.

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- 브라우저: [http://127.0.0.1:8000/](http://127.0.0.1:8000/) — 메인 `index.html`
- 정적 리소스: `http://127.0.0.1:8000/static/...`
- 헬스: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- OpenAPI 문서: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 디렉터리 개요

| 경로 | 설명 |
|------|------|
| `app/` | FastAPI 앱, RAG·기억 추출·STT·리포트 로직 |
| `static/` | 프론트 HTML/CSS/이미지 |
| `scripts/` | 시드·백필 등 운영/개발용 스크립트 |
| `docs/` | AI 에이전트용 기획·TDD 브리프 (`docs/README.md` 참고) |

## AI 네이티브 개발 문서

기능 설계·TDD를 에이전트에 맡길 때 쓰는 문서는 [`docs/README.md`](docs/README.md)를 참고하면 됩니다.
