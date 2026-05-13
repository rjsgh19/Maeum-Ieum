# DX_Team4 — AI 네이티브 개발 문서

이 폴더는 **백엔드·프론트엔드·DB**를 AI 에이전트와 함께 개발할 때, **구현 전 단계(기획·TDD)**까지 AI에게 위임할 수 있도록 정의한 문서 모음입니다.

| 파일 | 용도 |
|------|------|
| [`AI_AGENT_BRIEF.md`](./AI_AGENT_BRIEF.md) | **사용자가 채팅에 첨부하는 메인 문서** — AI 역할·목적·전체 워크플로 |
| [`AI_PLANNER_SPEC.md`](./AI_PLANNER_SPEC.md) | 구현 전 Planner 산출물 형식·체크리스트 |
| [`AI_TDD_SPEC.md`](./AI_TDD_SPEC.md) | TDD·검증 단계에서 AI가 따를 규칙 |

## 사용 방법 (요약)

1. `AI_AGENT_BRIEF.md`를 Cursor 등에 **컨텍스트로 첨부**한다.
2. 같은 메시지에 **자연어로** 원하는 기능·버그 수정·리팩터링을 적는다.
3. AI가 브리프에 따라 **Planner → TDD 설계 → 구현** 순으로 진행하도록 요청한다.

필요하면 `AI_PLANNER_SPEC.md` / `AI_TDD_SPEC.md`도 함께 첨부하면 산출물 형식이 더 안정적입니다.
