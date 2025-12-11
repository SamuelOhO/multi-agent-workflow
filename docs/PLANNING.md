# Agent Orchestrator System - 기획서

> Multi-Agent 협업 시스템: Agent를 쉽게 추가하고, 자동으로 다른 Agent들과 대화/협업할 수 있는 프레임워크

---

## 1. 프로젝트 개요

### 1.1 비전
여러 AI Agent가 서로 대화하고 협업하여 복잡한 작업을 수행하는 시스템.
새로운 Agent를 추가하면 별도 설정 없이 자동으로 기존 시스템과 통합되어 협업이 가능해야 함.

### 1.2 핵심 목표
1. **표준화된 Agent 인터페이스**: 모든 Agent가 동일한 방식으로 통신
2. **자동 발견(Auto-Discovery)**: 새 Agent 추가 시 자동 등록 및 통합
3. **유연한 대화 패턴**: 순차, 병렬, 토론 등 다양한 협업 방식 지원
4. **확장성**: Agent 수가 늘어나도 시스템 성능 유지

### 1.3 사용 시나리오

#### 시나리오 A: 소프트웨어 개발
```
[사용자] "웹 스크래퍼를 만들어줘"
    ↓
[Orchestrator] → [Research Agent] 기술 조사
    ↓
[Orchestrator] → [Architect Agent] 설계
    ↓
[Orchestrator] → [Coder Agent] 구현
    ↓
[Orchestrator] → [Reviewer Agent] 코드 리뷰
    ↓
[결과] 완성된 웹 스크래퍼 + 문서
```

#### 시나리오 B: 연구 및 분석
```
[사용자] "AI 트렌드 보고서 작성해줘"
    ↓
[Orchestrator] → [Search Agent] 병렬로 정보 수집
              → [Analysis Agent]
    ↓
[Orchestrator] → [Writer Agent] 보고서 작성
    ↓
[결과] 종합 보고서
```

---

## 2. 시스템 아키텍처

### 2.1 전체 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Orchestrator                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Gateway    │────▶│ Orchestrator │────▶│   Registry   │    │
│  │   (API)      │     │   (Core)     │◀────│   (Agents)   │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                              │                    │              │
│                              ▼                    ▼              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Message Bus                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│         │              │              │              │          │
│         ▼              ▼              ▼              ▼          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ Agent A  │   │ Agent B  │   │ Agent C  │   │ Agent N  │    │
│  │(Research)│   │ (Coder)  │   │(Reviewer)│   │  (...)   │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 핵심 컴포넌트

#### 2.2.1 Agent Registry (에이전트 등록소)
- **역할**: 모든 Agent의 메타데이터 관리
- **기능**:
  - Agent 등록/해제
  - Capability 기반 Agent 검색
  - Health check 및 상태 모니터링

```python
# Registry 데이터 구조
{
    "agent_id": "researcher_001",
    "name": "Research Agent",
    "capabilities": ["web_search", "data_extraction", "summarization"],
    "input_schema": {...},
    "output_schema": {...},
    "status": "active",
    "load": 0.3,  # 현재 부하
    "metadata": {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 4096,
        "avg_response_time_ms": 1500
    }
}
```

#### 2.2.2 Orchestrator (조정자)
- **역할**: 작업 분배 및 Agent 간 협업 조율
- **기능**:
  - Task 분석 및 Agent 선택
  - 대화 흐름 제어
  - 결과 병합 및 품질 검증

#### 2.2.3 Message Bus (메시지 버스)
- **역할**: Agent 간 비동기 통신 채널
- **기능**:
  - Pub/Sub 패턴 지원
  - 메시지 큐잉 및 전달 보장
  - 대화 이력 저장

#### 2.2.4 Gateway (게이트웨이)
- **역할**: 외부 인터페이스 제공
- **기능**:
  - REST API / WebSocket 엔드포인트
  - 인증/인가
  - Rate limiting

---

## 3. Agent 설계 규격

### 3.1 Agent 인터페이스 (필수 구현)

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pydantic import BaseModel

class AgentCapability(BaseModel):
    """Agent가 수행할 수 있는 작업 정의"""
    name: str                    # "web_search", "code_generation" 등
    description: str             # 상세 설명
    input_schema: Dict[str, Any] # 입력 형식
    output_schema: Dict[str, Any]# 출력 형식

class AgentConfig(BaseModel):
    """Agent 설정"""
    agent_id: str
    name: str
    description: str
    capabilities: List[AgentCapability]
    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 4096
    temperature: float = 0.7

class Message(BaseModel):
    """Agent 간 통신 메시지"""
    id: str
    sender_id: str
    recipient_id: str | None     # None이면 브로드캐스트
    message_type: str            # "task", "response", "query", "notification"
    content: Dict[str, Any]
    context: Dict[str, Any] = {} # 대화 맥락
    timestamp: str
    correlation_id: str          # 관련 대화 추적용

class BaseAgent(ABC):
    """모든 Agent가 구현해야 하는 기본 인터페이스"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.is_active = True

    @abstractmethod
    async def process(self, message: Message) -> Message:
        """메시지 처리 및 응답 생성"""
        pass

    @abstractmethod
    def can_handle(self, capability: str) -> bool:
        """특정 작업 처리 가능 여부"""
        pass

    def get_capabilities(self) -> List[str]:
        """지원하는 capability 목록 반환"""
        return [cap.name for cap in self.config.capabilities]

    async def health_check(self) -> Dict[str, Any]:
        """상태 체크"""
        return {
            "agent_id": self.config.agent_id,
            "status": "healthy" if self.is_active else "unhealthy",
            "capabilities": self.get_capabilities()
        }
```

### 3.2 Agent 설정 파일 형식 (YAML)

```yaml
# agents/researcher.yaml
agent_id: "researcher_001"
name: "Research Agent"
description: "웹 검색 및 정보 수집 전문 Agent"

capabilities:
  - name: "web_search"
    description: "인터넷에서 정보 검색"
    input_schema:
      type: "object"
      properties:
        query:
          type: "string"
          description: "검색 쿼리"
        max_results:
          type: "integer"
          default: 10
    output_schema:
      type: "object"
      properties:
        results:
          type: "array"
          items:
            type: "object"
            properties:
              title: { type: "string" }
              url: { type: "string" }
              snippet: { type: "string" }

  - name: "summarization"
    description: "텍스트 요약"
    input_schema:
      type: "object"
      properties:
        text:
          type: "string"
        max_length:
          type: "integer"
          default: 500
    output_schema:
      type: "object"
      properties:
        summary:
          type: "string"

model: "claude-haiku-4-5-20251001"
max_tokens: 4096
temperature: 0.3

system_prompt: |
  당신은 전문 리서치 Agent입니다.
  주어진 주제에 대해 깊이 있는 조사를 수행하고,
  정확하고 신뢰할 수 있는 정보를 제공합니다.

  응답 시 항상:
  - 출처를 명시하세요
  - 핵심 정보를 구조화하여 제시하세요
  - 불확실한 정보는 명확히 표시하세요
```

### 3.3 Agent 추가 방법

1. **YAML 설정 파일 작성**: `agents/` 폴더에 설정 파일 추가
2. **자동 로딩**: 시스템 시작 시 또는 Hot-reload로 자동 감지
3. **Registry 등록**: 자동으로 Registry에 등록되어 즉시 사용 가능

```python
# 새 Agent 추가는 설정 파일만 작성하면 끝
# agents/new_agent.yaml 생성 → 자동으로 시스템에 통합
```

---

## 4. 대화(Conversation) 패턴

### 4.1 지원하는 대화 패턴

#### 4.1.1 Sequential (순차적)
한 Agent의 출력이 다음 Agent의 입력이 됨

```
[Task] → [Agent A] → [Agent B] → [Agent C] → [Result]
```

**사용 사례**: 파이프라인 처리 (조사 → 설계 → 구현 → 검토)

#### 4.1.2 Parallel (병렬)
여러 Agent가 동시에 작업 수행 후 결과 병합

```
         ┌─→ [Agent A] ─┐
[Task] ──┼─→ [Agent B] ─┼─→ [Merge] → [Result]
         └─→ [Agent C] ─┘
```

**사용 사례**: 다중 소스 정보 수집, 병렬 분석

#### 4.1.3 Debate (토론/합의)
Agent들이 서로의 의견에 대해 토론 후 합의 도출

```
[Task] → [Agent A proposes] → [Agent B critiques]
                    ↓
       [Agent A responds] → [Agent B responds]
                    ↓
              [Consensus] → [Result]
```

**사용 사례**: 의사결정, 코드 리뷰, 문서 검토

#### 4.1.4 Hierarchical (계층적)
Supervisor Agent가 하위 Agent들을 관리

```
              [Supervisor]
                   │
     ┌─────────┼─────────┐
     ▼             ▼             ▼
[Worker A]   [Worker B]   [Worker C]
```

**사용 사례**: 대규모 작업 분할, 복잡한 프로젝트 관리

#### 4.1.5 Router (라우터)
조건에 따라 적절한 Agent로 라우팅

```
              [Router]
                 │
     ┌───────┼───────┐
     ▼           ▼           ▼
[Agent A]   [Agent B]   [Agent C]
(조건 A)     (조건 B)     (조건 C)
```

**사용 사례**: 질문 유형별 처리, 전문 분야별 할당

### 4.2 대화 설정 형식

```yaml
# conversations/software_development.yaml
name: "Software Development Pipeline"
description: "소프트웨어 개발 전체 과정을 수행"

pattern: "sequential"

stages:
  - name: "requirements"
    agent_capability: "requirements_analysis"
    description: "요구사항 분석"

  - name: "research"
    agent_capability: "web_search"
    description: "기술 조사"

  - name: "design"
    agent_capability: "architecture_design"
    description: "시스템 설계"

  - name: "implementation"
    agent_capability: "code_generation"
    description: "코드 구현"

  - name: "review"
    agent_capability: "code_review"
    description: "코드 리뷰"
    parallel_with: "testing"  # 리뷰와 테스트 병렬 실행

  - name: "testing"
    agent_capability: "test_generation"
    description: "테스트 작성"

merge_strategy: "structured"  # 결과 병합 방식
max_iterations: 3             # 최대 반복 횟수 (토론 패턴용)
timeout_seconds: 300          # 전체 타임아웃
```

---

## 5. 기술 스택

### 5.1 권장 스택

| 구분 | 기술 | 이유 |
|------|------|------|
| **언어** | Python 3.11+ | 풍부한 AI 생태계, async 지원 |
| **웹 프레임워크** | FastAPI | 비동기 지원, 자동 문서화 |
| **메시지 큐** | Redis Streams 또는 RabbitMQ | 가벼움 / 신뢰성 |
| **데이터 검증** | Pydantic v2 | 타입 안전성, 직렬화 |
| **AI API** | Anthropic Claude API | 성능, 안정성 |
| **설정 관리** | YAML + Pydantic | 가독성, 검증 |
| **로깅** | structlog | 구조화된 로깅 |
| **테스트** | pytest + pytest-asyncio | 비동기 테스트 지원 |

### 5.2 프로젝트 구조

```
agent-orchestrator/
├── src/
│   ├── __init__.py
│   ├── main.py                 # 애플리케이션 진입점
│   │
│   ├── core/                   # 핵심 로직
│   │   ├── __init__.py
│   │   ├── orchestrator.py     # Orchestrator 클래스
│   │   ├── registry.py         # Agent Registry
│   │   ├── message_bus.py      # 메시지 버스
│   │   └── conversation.py     # 대화 관리
│   │
│   ├── agents/                 # Agent 구현
│   │   ├── __init__.py
│   │   ├── base.py             # BaseAgent 추상 클래스
│   │   ├── loader.py           # Agent 동적 로딩
│   │   └── implementations/    # 구체적인 Agent 구현
│   │       ├── __init__.py
│   │       ├── researcher.py
│   │       ├── coder.py
│   │       └── reviewer.py
│   │
│   ├── patterns/               # 대화 패턴
│   │   ├── __init__.py
│   │   ├── base.py             # 기본 패턴
│   │   ├── sequential.py
│   │   ├── parallel.py
│   │   ├── debate.py
│   │   └── router.py
│   │
│   ├── api/                    # REST API
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── models/                 # 데이터 모델
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── message.py
│   │   └── conversation.py
│   │
│   └── utils/                  # 유틸리티
│       ├── __init__.py
│       ├── config.py
│       └── logging.py
│
├── configs/                    # 설정 파일
│   ├── agents/                 # Agent 설정
│   │   ├── researcher.yaml
│   │   ├── coder.yaml
│   │   └── reviewer.yaml
│   ├── conversations/          # 대화 패턴 설정
│   │   └── dev_pipeline.yaml
│   └── app.yaml                # 앱 전체 설정
│
├── tests/                      # 테스트
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_registry.py
│   │   ├── test_orchestrator.py
│   │   └── test_patterns.py
│   └── integration/
│       └── test_conversation.py
│
├── examples/                   # 사용 예제
│   ├── simple_pipeline.py
│   └── custom_agent.py
│
├── scripts/                    # 유틸리티 스크립트
│   └── create_agent.py         # Agent 템플릿 생성
│
├── pyproject.toml              # 프로젝트 설정
├── requirements.txt            # 의존성
├── README.md                   # 프로젝트 설명
├── PLANNING.md                 # 이 문서
├── DEVELOPMENT_CHECKLIST.md    # 개발 체크리스트
└── .env.example                # 환경 변수 예시
```

---

## 6. API 설계

### 6.1 REST API 엔드포인트

#### Agent 관리
```
GET    /api/v1/agents              # Agent 목록 조회
GET    /api/v1/agents/{id}         # Agent 상세 조회
POST   /api/v1/agents              # Agent 등록 (동적)
DELETE /api/v1/agents/{id}         # Agent 제거
GET    /api/v1/agents/{id}/health  # Agent 상태 체크
```

#### 작업 실행
```
POST   /api/v1/tasks               # 새 작업 생성
GET    /api/v1/tasks/{id}          # 작업 상태 조회
GET    /api/v1/tasks/{id}/result   # 작업 결과 조회
DELETE /api/v1/tasks/{id}          # 작업 취소
```

#### 대화
```
POST   /api/v1/conversations                    # 대화 시작
GET    /api/v1/conversations/{id}               # 대화 상태
GET    /api/v1/conversations/{id}/messages      # 메시지 이력
POST   /api/v1/conversations/{id}/messages      # 메시지 추가
```

### 6.2 WebSocket (실시간)
```
WS /ws/conversations/{id}    # 실시간 대화 스트림
WS /ws/tasks/{id}            # 작업 진행 상황 스트림
```

### 6.3 API 응답 형식

```json
{
    "success": true,
    "data": {
        "task_id": "task_abc123",
        "status": "in_progress",
        "agents_involved": ["researcher_001", "coder_001"],
        "progress": {
            "current_stage": "research",
            "completed_stages": ["requirements"],
            "remaining_stages": ["design", "implementation", "review"]
        },
        "intermediate_results": [
            {
                "stage": "requirements",
                "agent_id": "researcher_001",
                "output": {...}
            }
        ]
    },
    "metadata": {
        "timestamp": "2025-11-28T12:00:00Z",
        "request_id": "req_xyz789"
    }
}
```

---

## 7. 보안 고려사항

### 7.1 인증/인가
- API Key 기반 인증
- JWT 토큰 지원 (선택)
- Agent별 권한 제어

### 7.2 데이터 보호
- 민감 정보 마스킹
- 대화 이력 암호화 (선택)
- Rate limiting

### 7.3 Agent 격리
- 각 Agent의 접근 범위 제한
- 리소스 사용량 제한
- 샌드박스 실행 (코드 실행 Agent)

---

## 8. 모니터링 및 관찰성

### 8.1 로깅
- 구조화된 JSON 로깅
- 상관 ID(Correlation ID)로 대화 추적
- 로그 레벨: DEBUG, INFO, WARNING, ERROR

### 8.2 메트릭
- Agent별 응답 시간
- 성공/실패율
- 토큰 사용량
- 동시 작업 수

### 8.3 Langfuse 연동 (LLM 관측성)

Agent 간 대화 흐름을 추적하고 시각화하기 위해 [Langfuse](https://langfuse.com/)를 사용합니다.

#### 8.3.1 Langfuse 선택 이유
| 항목 | 설명 |
|------|------|
| **오픈소스** | MIT 라이선스, Self-hosted 가능 |
| **프레임워크 무관** | REST API로 어떤 시스템이든 연동 가능 |
| **트레이싱** | Agent 호출 순서, 소요 시간, 입출력 추적 |
| **비용 추적** | 토큰 사용량, API 비용 자동 계산 |
| **프롬프트 관리** | 버전 관리, A/B 테스트 지원 |
| **팀 협업** | 멀티유저, 프로젝트 관리 |

#### 8.3.2 연동 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Orchestrator                        │
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐  │
│  │ Message Bus  │────▶│ Observability│────▶│  Langfuse   │  │
│  │              │     │   Module     │     │   Server    │  │
│  └──────────────┘     └──────────────┘     └─────────────┘  │
│         │                    │                    │          │
│         ▼                    ▼                    ▼          │
│  ┌──────────┐         ┌──────────┐         ┌──────────┐     │
│  │ Agent A  │         │  Trace   │         │ Dashboard │     │
│  │ Agent B  │         │  Spans   │         │ 시각화    │     │
│  │ Agent C  │         │  Events  │         │ 분석      │     │
│  └──────────┘         └──────────┘         └──────────┘     │
└─────────────────────────────────────────────────────────────┘
```

#### 8.3.3 추적 데이터 매핑

```python
# Message → Langfuse Span 매핑
Message.correlation_id  →  trace_id      # 대화 전체 추적
Message.id              →  span_id       # 개별 메시지 추적
Message.sender_id       →  metadata      # Agent 정보
Message.recipient_id    →  metadata      # Agent 정보
Message.content         →  input/output  # 메시지 내용
Message.timestamp       →  timestamp     # 시간 정보

# Conversation → Langfuse Trace
Conversation.id         →  trace_id
Conversation.name       →  trace_name
Conversation.stages     →  spans (계층 구조)
```

#### 8.3.4 구현 위치
- `src/utils/observability.py` - Langfuse 클라이언트 래퍼
- `src/core/message_bus.py` - 메시지 발행 시 자동 추적
- `src/agents/base.py` - Agent 호출 시 span 생성

### 8.4 대시보드 (향후)
- 실시간 Agent 상태
- 대화 흐름 시각화 (Langfuse UI 활용)
- 성능 그래프

---

## 9. 확장 계획

### Phase 1: MVP (현재)
- [x] 기본 아키텍처 설계
- [ ] Core 컴포넌트 구현
- [ ] 기본 Agent 3개 (Researcher, Coder, Reviewer)
- [ ] Sequential/Parallel 패턴
- [ ] REST API

### Phase 2: 고급 기능
- [ ] Debate 패턴
- [ ] WebSocket 실시간 스트림
- [ ] Agent 동적 로딩 (Hot-reload)
- [ ] 대화 이력 저장

### Phase 3: 프로덕션
- [ ] 분산 처리 지원
- [ ] 고가용성 구성
- [ ] 관리 대시보드
- [ ] 플러그인 시스템

---

## 10. 참고 자료

### 10.1 관련 프로젝트
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agent 워크플로우
- [AutoGen](https://github.com/microsoft/autogen) - Multi-Agent 프레임워크
- [CrewAI](https://github.com/joaomdmoura/crewAI) - Role-based Agent 협업

### 10.2 문서
- [Anthropic Claude API](https://docs.anthropic.com/)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [Pydantic 문서](https://docs.pydantic.dev/)
- [Langfuse 문서](https://langfuse.com/docs) - LLM 관측성

---

*마지막 업데이트: 2025-11-28*
