# Agent Orchestrator - 개발 체크리스트

> PLANNING.md의 설계를 구현하기 위한 단계별 TODO 리스트
> 각 항목 완료 시 `[ ]`를 `[x]`로 변경하세요

---

## Phase 1: 프로젝트 초기 설정

### 1.1 프로젝트 구조 생성
- [x] 기본 디렉토리 구조 생성
  ```
  mkdir -p src/{core,agents/implementations,patterns,api,models,utils}
  mkdir -p configs/{agents,conversations}
  mkdir -p tests/{unit,integration}
  mkdir -p examples scripts
  ```
- [x] `pyproject.toml` 생성 (프로젝트 메타데이터 및 의존성)
- [x] `requirements.txt` 생성
- [x] `.env.example` 생성 (환경 변수 템플릿)
- [x] `.gitignore` 생성
- [x] `README.md` 기본 내용 작성

### 1.2 의존성 설치
- [x] Python 가상환경 생성 (`python -m venv venv`)
- [x] 핵심 패키지 설치
  - [x] `fastapi` (0.122.0)
  - [x] `uvicorn[standard]` (0.38.0)
  - [x] `pydantic>=2.0` (2.12.5)
  - [x] `anthropic` (0.75.0)
  - [x] `pyyaml` (6.0.3)
  - [x] `structlog` (25.5.0)
  - [x] `python-dotenv` (1.2.1)
  - [x] `httpx` (0.28.1)
  - [ ] `redis` (선택)
- [x] 개발 패키지 설치
  - [x] `pytest` (9.0.1)
  - [x] `pytest-asyncio` (1.3.0)
  - [x] `pytest-cov` (7.0.0)
  - [x] `black` (25.11.0)
  - [x] `ruff` (0.14.6)
  - [x] `mypy` (1.18.2)
  - [x] `pre-commit` (4.5.0)

### 📋 Phase 1 이슈 및 참고사항

#### ✅ 완료된 항목
| 항목 | 파일 | 비고 |
|------|------|------|
| 디렉토리 구조 | `src/`, `configs/`, `tests/`, `examples/`, `scripts/` | 모든 `__init__.py` 포함 |
| pyproject.toml | `pyproject.toml` | black, ruff, mypy, pytest 설정 포함 |
| requirements.txt | `requirements.txt` | 핵심 + 개발 의존성 |
| 환경변수 템플릿 | `.env.example` | API 키, 앱 설정, 타임아웃 등 |
| Git 설정 | `.gitignore` | Python, IDE, 환경 파일 제외 |
| 문서 | `README.md` | 빠른 시작 가이드 포함 |
| 가상환경 | `venv/` | Python 가상환경 생성 완료 |
| 핵심 패키지 | pip list 참조 | fastapi, uvicorn, pydantic, anthropic 등 |
| 개발 패키지 | pip list 참조 | pytest, black, ruff, mypy, pre-commit 등 |

#### ⚠️ 주의사항
1. **가상환경 활성화 방법**
   ```bash
   venv\Scripts\activate     # Windows
   source venv/bin/activate  # Linux/Mac
   ```

2. **Anthropic API 키 필수**
   - `.env` 파일 생성 후 `ANTHROPIC_API_KEY` 설정 필요
   - `.env.example` 파일을 복사하여 `.env` 파일 생성

3. **Python 버전**
   - Python 3.11 이상 필요 (3.11+ 타입 힌팅 문법 사용)

4. **pip 업데이트 권장**
   ```bash
   python -m pip install --upgrade pip
   ```
   - 현재 pip 24.2 → 25.3 업데이트 가능

#### 🔧 다음 단계 준비사항
- Phase 2 (데이터 모델 구현) 진행 준비 완료
- `configs/app.yaml` 파일은 Phase 7에서 생성 예정

#### 📦 설치된 패키지 버전 (2025-11-28)
| 패키지 | 버전 | 용도 |
|--------|------|------|
| fastapi | 0.122.0 | 웹 프레임워크 |
| uvicorn | 0.38.0 | ASGI 서버 |
| pydantic | 2.12.5 | 데이터 검증 |
| anthropic | 0.75.0 | Claude API |
| structlog | 25.5.0 | 구조화 로깅 |
| httpx | 0.28.1 | HTTP 클라이언트 |
| pytest | 9.0.1 | 테스트 프레임워크 |
| black | 25.11.0 | 코드 포매터 |
| ruff | 0.14.6 | 린터 |
| mypy | 1.18.2 | 타입 체커 |

---

## Phase 2: 데이터 모델 구현

### 2.1 기본 모델 (`src/models/`)
- [x] `src/models/__init__.py` 생성
- [x] `src/models/agent.py`
  - [x] `AgentCapability` 모델 정의
  - [x] `AgentConfig` 모델 정의
  - [x] `AgentStatus` Enum 정의
  - [x] `AgentInfo` 모델 정의 (런타임 정보)
- [x] `src/models/message.py`
  - [x] `MessageType` Enum 정의 (task, response, query, notification, error, system)
  - [x] `Message` 모델 정의
  - [x] `MessageContext` 모델 정의
- [x] `src/models/conversation.py`
  - [x] `ConversationStatus` Enum 정의
  - [x] `ConversationPattern` Enum 정의
  - [x] `ConversationStage` 모델 정의
  - [x] `Conversation` 모델 정의
  - [x] `ConversationResult` 모델 정의
- [x] 모델 단위 테스트 작성 (`tests/unit/test_models.py`) - 36개 테스트 통과

### 📋 Phase 2 이슈 및 참고사항

#### ✅ 구현된 모델
| 파일 | 클래스/Enum | 설명 |
|------|-------------|------|
| `agent.py` | `AgentStatus` | Agent 상태 (active, inactive, busy, error, starting, stopping) |
| `agent.py` | `AgentCapability` | Agent 능력 정의 (name, description, input/output schema) |
| `agent.py` | `AgentConfig` | Agent 설정 (id, name, capabilities, model, temperature 등) |
| `agent.py` | `AgentInfo` | Agent 런타임 정보 (status, load, metadata) |
| `message.py` | `MessageType` | 메시지 유형 (task, response, query, notification, error, system) |
| `message.py` | `MessageContext` | 메시지 컨텍스트 (conversation_id, parent_message_id, history) |
| `message.py` | `Message` | Agent 간 통신 메시지 (sender, recipient, content, correlation_id) |
| `conversation.py` | `ConversationStatus` | 대화 상태 (pending, in_progress, completed, failed, cancelled, paused) |
| `conversation.py` | `ConversationPattern` | 대화 패턴 (sequential, parallel, debate, hierarchical, router) |
| `conversation.py` | `ConversationStage` | 대화 단계 (name, agent_capability, status, input/output) |
| `conversation.py` | `Conversation` | 대화 세션 (stages, messages, pattern, timeout) |
| `conversation.py` | `ConversationResult` | 대화 결과 (status, output, duration, intermediate_results) |

#### 🔧 구현 특이사항
1. **Message 헬퍼 메서드**: `create_task()`, `create_response()`, `create_error()` 제공
2. **자동 생성 필드**: `id`, `timestamp`, `correlation_id`는 자동 생성
3. **유효성 검증**: Pydantic v2 사용, temperature(0.0~2.0), load(0.0~1.0) 범위 검증
4. **from_config/from_conversation**: 편의 메서드로 객체 변환 지원

---

## Phase 3: 핵심 컴포넌트 구현

### 3.1 Agent Registry (`src/core/registry.py`)
- [x] `AgentRegistry` 클래스 생성
  - [x] `register(agent)` 메서드 - Agent 등록
  - [x] `unregister(agent_id)` 메서드 - Agent 제거
  - [x] `get(agent_id)` 메서드 - Agent 조회
  - [x] `list_all()` 메서드 - 전체 Agent 목록
  - [x] `find_by_capability(capability)` 메서드 - 기능별 검색
  - [x] `health_check_all()` 메서드 - 전체 상태 체크
- [x] Registry 단위 테스트 (`tests/unit/test_registry.py`) - 15개 테스트 통과

### 3.2 Message Bus (`src/core/message_bus.py`)
- [x] `MessageBus` 인터페이스 정의
  - [x] `publish(message)` 메서드
  - [x] `subscribe(agent_id, callback)` 메서드
  - [x] `unsubscribe(agent_id)` 메서드
- [x] `InMemoryMessageBus` 구현 (단순 버전)
  - [x] 메시지 큐 관리
  - [x] 메시지 라우팅
  - [x] 대화 이력 저장
- [ ] (선택) `RedisMessageBus` 구현 (프로덕션 버전)
- [x] Message Bus 단위 테스트 (`tests/unit/test_message_bus.py`) - 14개 테스트 통과

### 3.3 Conversation Manager (`src/core/conversation.py`)
- [x] `ConversationManager` 클래스 생성
  - [x] `create(task, pattern)` 메서드 - 대화 생성
  - [x] `get(conversation_id)` 메서드 - 대화 조회
  - [x] `get_messages(conversation_id)` 메서드 - 메시지 이력
  - [x] `add_message(conversation_id, message)` 메서드
  - [x] `update_status(conversation_id, status)` 메서드
- [x] Conversation Manager 테스트 (`tests/unit/test_conversation_manager.py`) - 22개 테스트 통과

### 3.4 Orchestrator (`src/core/orchestrator.py`)
- [x] `Orchestrator` 클래스 생성
  - [x] 의존성 주입 (Registry, MessageBus, ConversationManager)
  - [x] `execute(task, pattern)` 메서드 - 작업 실행
  - [x] `_analyze_and_route(task)` 메서드 - 작업 분석 및 라우팅
  - [x] `_select_agent(stage)` 메서드 - Agent 선택
  - [x] `_execute_sequential/parallel/router/debate` 패턴별 실행 메서드
  - [x] `_execute_single_stage(conversation, stage)` 메서드 - 단일 스테이지 실행
  - [x] `cancel(conversation_id)` 메서드 - 작업 취소
- [x] Orchestrator 단위 테스트 (`tests/unit/test_orchestrator.py`) - 12개 테스트 통과

### 📋 Phase 3 이슈 및 참고사항

#### ✅ 구현된 컴포넌트
| 컴포넌트 | 파일 | 테스트 수 | 설명 |
|---------|------|----------|------|
| AgentRegistry | `src/core/registry.py` | 15 | Agent 등록/조회/검색, 상태 관리 |
| MessageBus | `src/core/message_bus.py` | 14 | Pub/Sub 메시징, 이력 저장, 브로드캐스트 |
| ConversationManager | `src/core/conversation.py` | 22 | 대화 생성/조회/상태 관리, 스테이지 관리 |
| Orchestrator | `src/core/orchestrator.py` | 12 | 패턴 실행, Agent 선택, 결과 병합 |

#### 🔧 구현 특이사항
1. **비동기 지원**: 모든 메서드가 `async/await` 기반으로 구현됨
2. **Thread-Safe**: `asyncio.Lock` 사용하여 동시성 제어
3. **패턴 실행**: Sequential, Parallel, Router, Debate 4가지 패턴 지원
4. **에러 처리**: 커스텀 예외 클래스 정의 (AgentNotFoundError, ConversationNotFoundError 등)
5. **Protocol 기반**: AgentProtocol 정의로 타입 안전성 확보

#### ⚠️ 참고사항
1. **RedisMessageBus**: 프로덕션 버전은 아직 미구현 (placeholder만 존재)
2. **Debate 패턴**: Agent 응답의 `result` 필드 내에 `consensus`, `proposal` 포함 필요
3. **테스트 총합**: Phase 3 단위 테스트 63개 전체 통과

---

## Phase 4: Agent 구현

### 4.1 Base Agent (`src/agents/base.py`)
- [x] `BaseAgent` 추상 클래스 구현
  - [x] `__init__(config)` - 초기화
  - [x] `process(message)` 추상 메서드
  - [x] `can_handle(capability)` 메서드
  - [x] `get_capabilities()` 메서드
  - [x] `health_check()` 메서드
  - [x] `_call_llm(prompt, context)` 헬퍼 메서드 (Claude API 호출)
- [x] Base Agent 테스트 (`tests/unit/test_base_agent.py`) - 25개 테스트 통과

### 4.2 Agent Loader (`src/agents/loader.py`)
- [x] `AgentLoader` 클래스 구현
  - [x] `load_from_yaml(path)` 메서드 - YAML에서 Agent 로드
  - [x] `load_all_from_directory(dir_path)` 메서드 - 디렉토리 전체 로드
  - [x] `create_agent(config)` 메서드 - Agent 인스턴스 생성
  - [x] YAML 스키마 검증
- [x] Agent Loader 테스트 (`tests/unit/test_agent_loader.py`) - 33개 테스트 통과

### 4.3 구체적 Agent 구현

#### Research Agent (`src/agents/implementations/researcher.py`)
- [x] `ResearchAgent` 클래스 구현
  - [x] `process()` 구현
  - [x] 웹 검색 기능 (LLM 시뮬레이션)
  - [x] 정보 요약 기능
  - [x] 출력 구조화
- [x] `configs/agents/researcher.yaml` 설정 파일 작성
- [x] Research Agent 테스트 - 7개 테스트 통과

#### Coder Agent (`src/agents/implementations/coder.py`)
- [x] `CoderAgent` 클래스 구현
  - [x] `process()` 구현
  - [x] 코드 생성 기능
  - [x] 코드 수정 기능
  - [x] 코드 설명/디버깅 기능
- [x] `configs/agents/coder.yaml` 설정 파일 작성
- [x] Coder Agent 테스트 - 6개 테스트 통과

#### Reviewer Agent (`src/agents/implementations/reviewer.py`)
- [x] `ReviewerAgent` 클래스 구현
  - [x] `process()` 구현
  - [x] 코드 리뷰 기능
  - [x] 보안/성능/스타일 리뷰
  - [x] 개선 제안 생성
- [x] `configs/agents/reviewer.yaml` 설정 파일 작성
- [x] Reviewer Agent 테스트 - 6개 테스트 통과

### 📋 Phase 4 이슈 및 참고사항

#### ✅ 구현된 컴포넌트
| 컴포넌트 | 파일 | 테스트 수 | 설명 |
|---------|------|----------|------|
| BaseAgent | `src/agents/base.py` | 25 | 추상 기본 클래스, LLM 호출, 상태 관리 |
| SimpleAgent | `src/agents/base.py` | - | BaseAgent의 기본 구현 |
| AgentLoader | `src/agents/loader.py` | 33 | YAML 로딩, 동적 Agent 생성 |
| ResearchAgent | `src/agents/implementations/researcher.py` | 7 | 웹 검색, 요약, 데이터 추출 |
| CoderAgent | `src/agents/implementations/coder.py` | 6 | 코드 생성, 수정, 설명, 디버깅 |
| ReviewerAgent | `src/agents/implementations/reviewer.py` | 6 | 코드/보안/성능/스타일 리뷰 |

#### 🔧 구현 특이사항
1. **LLM 호출**: `_call_llm()` 메서드로 Anthropic Claude API 호출
2. **상태 관리**: `activate()`, `deactivate()`, `set_busy()`, `set_error()` 메서드 제공
3. **커스텀 Agent 로딩**: `agent_class` 필드로 사용자 정의 Agent 클래스 지정 가능
4. **YAML 설정**: 각 Agent는 `configs/agents/` 디렉토리에 YAML 파일로 설정
5. **에러 핸들링**: `AgentError`, `LLMError`, `AgentLoadError`, `AgentConfigError` 예외 클래스

#### ⚠️ 참고사항
1. **API 키 필수**: `ANTHROPIC_API_KEY` 환경 변수 설정 필요
2. **테스트 총합**: Phase 4 단위 테스트 80개 전체 통과 (총 179개)
3. **ruff/mypy 통과**: 코드 품질 검사 통과

---

## Phase 5: 대화 패턴 구현

### 5.1 기본 패턴 (`src/patterns/base.py`)
- [x] `BasePattern` 추상 클래스 정의
  - [x] `execute(conversation)` 추상 메서드
  - [x] `validate_conversation(conversation)` 메서드
  - [x] `_select_agent(stage)` 헬퍼 메서드
  - [x] `_execute_single_stage(conversation, stage)` 헬퍼 메서드
  - [x] `_mark_stage_started/completed/failed` 헬퍼 메서드
- [x] 패턴 팩토리 함수 (`get_pattern_class`, `create_pattern`)
- [x] 커스텀 예외 클래스 (`PatternError`, `NoAgentAvailableError`, `StageExecutionError`, `PatternValidationError`)

### 5.2 Sequential Pattern (`src/patterns/sequential.py`)
- [x] `SequentialPattern` 클래스 구현
  - [x] 순차적 Agent 실행
  - [x] 이전 결과를 다음 Agent에 전달
  - [x] 에러 핸들링 (중간 실패 시 전체 실패)
- [x] Sequential 패턴 테스트 (`tests/unit/test_patterns.py`)

### 5.3 Parallel Pattern (`src/patterns/parallel.py`)
- [x] `ParallelPattern` 클래스 구현
  - [x] `asyncio.gather`로 병렬 실행
  - [x] 결과 병합 전략 (`MergeStrategy`: dict, list, flatten)
  - [x] 부분 실패 처리 (모든 에러 수집 후 실패)
- [x] Parallel 패턴 테스트

### 5.4 Router Pattern (`src/patterns/router.py`)
- [x] `RouterPattern` 클래스 구현
  - [x] 라우팅 조건 정의 (`add_routing_condition`)
  - [x] 조건 평가 로직 (명시적 route → 커스텀 조건 → 키워드 매칭 → 기본 라우트)
  - [x] 기본 라우트 처리 (`set_default_route`)
  - [x] 키워드 기반 자동 라우팅
- [x] Router 패턴 테스트

### 5.5 Debate Pattern (`src/patterns/debate.py`)
- [x] `DebatePattern` 클래스 구현
  - [x] 라운드 기반 토론 (`DebateRound` 유틸리티 클래스)
  - [x] 합의 도출 로직 (`consensus: True` 시그널)
  - [x] 최대 라운드 제한 (`max_iterations`)
  - [x] 최종 결론 생성
  - [x] 모더레이터 옵션 (`execute_with_moderator`)
- [x] Debate 패턴 테스트

### 📋 Phase 5 이슈 및 참고사항

#### ✅ 구현된 컴포넌트
| 컴포넌트 | 파일 | 설명 |
|---------|------|------|
| BasePattern | `src/patterns/base.py` | 추상 기본 클래스, 공통 헬퍼 메서드 |
| SequentialPattern | `src/patterns/sequential.py` | 순차 실행, 출력→입력 체이닝 |
| ParallelPattern | `src/patterns/parallel.py` | 병렬 실행, 다중 병합 전략 |
| RouterPattern | `src/patterns/router.py` | 조건 기반 라우팅, 키워드 매칭 |
| DebatePattern | `src/patterns/debate.py` | 토론/합의, 라운드 기반, 모더레이터 |
| HierarchicalPattern | `src/patterns/hierarchical.py` | Supervisor-Worker 계층 구조 |

#### 🔧 구현 특이사항
1. **패턴 검증**: 각 패턴은 `validate_conversation()` 메서드로 사전 검증
2. **에러 처리**: 커스텀 예외 클래스로 상세한 에러 정보 제공
3. **팩토리 패턴**: `create_pattern()` 함수로 패턴 인스턴스 동적 생성
4. **병합 전략**: ParallelPattern에서 dict/list/flatten 3가지 병합 방식 지원
5. **Debate 모더레이터**: 선택적 모더레이터 Agent로 토론 진행 가이드 가능

### 5.6 Hierarchical Pattern (`src/patterns/hierarchical.py`)
- [x] `HierarchicalPattern` 클래스 구현
  - [x] Supervisor Agent가 Worker Agent들 관리
  - [x] 작업 위임 (`delegations` 구조)
  - [x] 병렬/순차 Worker 실행 모드
  - [x] 결과 집계 (Supervisor가 최종 결과 생성)
  - [x] `delegate_to_all` 옵션으로 전체 Worker 위임
- [x] Hierarchical 패턴 테스트 - 7개 테스트 통과

#### ⚠️ 참고사항
1. **테스트 총합**: Phase 5 단위 테스트 43개 전체 통과 (총 222개)
2. **ruff/mypy 통과**: 코드 품질 검사 통과

---

## Phase 6: API 구현

### 6.1 API 스키마 (`src/api/schemas.py`)
- [x] Request 스키마 정의
  - [x] `CreateTaskRequest`
  - [x] `CreateConversationRequest`
  - [x] `RegisterAgentRequest`
  - [x] `AddMessageRequest`
  - [x] `ConversationStageRequest`
- [x] Response 스키마 정의
  - [x] `TaskStatusResponse`
  - [x] `TaskResultResponse`
  - [x] `AgentResponse`
  - [x] `AgentHealthResponse`
  - [x] `ConversationResponse`
  - [x] `MessageResponse`
  - [x] `ErrorResponse`
  - [x] `APIResponse` (표준 응답 형식)
  - [x] `PaginatedResponse`

### 6.2 API 라우트 (`src/api/routes.py`)
- [x] Agent 엔드포인트
  - [x] `GET /api/v1/agents` - 목록 조회
  - [x] `GET /api/v1/agents/{id}` - 상세 조회
  - [x] `POST /api/v1/agents` - 등록
  - [x] `DELETE /api/v1/agents/{id}` - 제거
  - [x] `GET /api/v1/agents/{id}/health` - 상태 체크
- [x] Task 엔드포인트
  - [x] `POST /api/v1/tasks` - 작업 생성 (백그라운드 실행)
  - [x] `GET /api/v1/tasks/{id}` - 상태 조회
  - [x] `GET /api/v1/tasks/{id}/result` - 결과 조회
  - [x] `DELETE /api/v1/tasks/{id}` - 취소
- [x] Conversation 엔드포인트
  - [x] `POST /api/v1/conversations` - 대화 시작
  - [x] `GET /api/v1/conversations/{id}` - 상태 조회
  - [x] `GET /api/v1/conversations/{id}/messages` - 메시지 조회
  - [x] `POST /api/v1/conversations/{id}/messages` - 메시지 추가
- [x] Health Check 엔드포인트
  - [x] `GET /api/v1/health` - 시스템 상태 체크

### 6.3 API 통합 테스트
- [x] `tests/integration/test_api.py` 작성
- [x] Agent API 테스트 (9개 테스트)
- [x] Task API 테스트 (4개 테스트)
- [x] Conversation API 테스트 (5개 테스트)
- [x] Schema Validation 테스트 (4개 테스트)
- [x] Response Format 테스트 (2개 테스트)

### 📋 Phase 6 이슈 및 참고사항

#### ✅ 구현된 컴포넌트
| 컴포넌트 | 파일 | 설명 |
|---------|------|------|
| API 스키마 | `src/api/schemas.py` | Pydantic 모델 기반 Request/Response 스키마 |
| API 라우트 | `src/api/routes.py` | FastAPI 라우터, 의존성 주입 |
| 통합 테스트 | `tests/integration/test_api.py` | 24개 테스트 |
| 테스트 fixtures | `tests/conftest.py` | 공통 테스트 fixtures |

#### 🔧 구현 특이사항
1. **표준 응답 형식**: 모든 API는 `APIResponse` 형식으로 응답 (`success`, `data`, `error`, `metadata`)
2. **백그라운드 작업**: Task 생성 시 `BackgroundTasks`로 비동기 실행
3. **의존성 주입**: `init_dependencies()`로 전역 의존성 설정
4. **에러 처리**: HTTP 상태 코드별 적절한 에러 응답 (404, 409, 422)
5. **SimpleAgent 사용**: 동적 Agent 등록 시 `SimpleAgent` 인스턴스 생성

#### ⚠️ 참고사항
1. **테스트 총합**: Phase 6 통합 테스트 24개 전체 통과 (총 246개)
2. **ruff/mypy 통과**: 코드 품질 검사 통과
3. **메인 앱 미구현**: `src/main.py`는 Phase 8에서 구현 예정

---

## Phase 7: 유틸리티 및 설정

### 7.1 설정 관리 (`src/utils/config.py`)
- [x] `AppConfig` 클래스 구현
  - [x] 환경 변수 로딩 (`from_env()`, `_apply_env_overrides()`)
  - [x] YAML 설정 로딩 (`from_yaml()`, `load()`)
  - [x] 설정 검증 (Pydantic validators)
- [x] `configs/app.yaml` 기본 설정 파일 작성

### 7.2 로깅 (`src/utils/logging.py`)
- [x] structlog 설정 (`setup_logging()`)
- [x] JSON 포맷터 설정 (`json_format=True`)
- [x] 로그 레벨 설정 (`level` 파라미터)
- [x] Correlation ID 지원 (`set_correlation_id()`, `get_correlation_id()`, `clear_correlation_id()`)

### 7.3 에러 핸들링
- [x] 커스텀 Exception 클래스 정의 (`src/utils/exceptions.py`)
- [x] 글로벌 에러 핸들러 구현 (`src/utils/error_handlers.py`)

### 7.4 Langfuse 연동 (`src/utils/observability.py`)
- [x] `langfuse` 패키지 설치 (`requirements.txt`에 추가)
- [x] `LangfuseClient` 래퍼 클래스 구현
  - [x] 환경 변수에서 설정 로딩 (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)
  - [x] `start_trace(trace_id, name, metadata)` 메서드
  - [x] `start_span(span_id, trace_id, name, input_data, metadata)` 메서드
  - [x] `end_span(span_id, output, status)` 메서드
  - [x] `log_message(trace_id, message_id, ...)` 메서드
  - [x] `log_generation()` 메서드 (LLM 호출 추적)
  - [x] `trace_context()`, `span_context()` 컨텍스트 매니저
- [ ] Message Bus 연동 (선택 - Phase 8에서 통합 가능)
  - [ ] `publish()` 시 자동 span 생성
  - [ ] Agent 응답 시 span 종료
- [ ] BaseAgent 연동 (선택 - Phase 8에서 통합 가능)
  - [ ] `process()` 호출 시 자동 추적
  - [ ] LLM 호출 비용/토큰 기록
- [x] 테스트
  - [x] Mock을 사용한 단위 테스트 (56개 테스트)
  - [x] Langfuse 비활성화 시 graceful 처리 확인

### 7.5 에러 응답
- [x] 에러 응답 형식 통일 (`create_error_response()`, `APIResponse`)

### 📋 Phase 7 이슈 및 참고사항

#### ✅ 구현된 컴포넌트
| 컴포넌트 | 파일 | 테스트 수 | 설명 |
|---------|------|----------|------|
| AppConfig | `src/utils/config.py` | 18 | 환경변수/YAML 설정 로딩, 검증 |
| Logging | `src/utils/logging.py` | 11 | structlog 기반 구조화 로깅 |
| Exceptions | `src/utils/exceptions.py` | 20 | 계층적 예외 클래스 |
| Error Handlers | `src/utils/error_handlers.py` | - | FastAPI 글로벌 에러 핸들러 |
| Observability | `src/utils/observability.py` | 7 | Langfuse 연동, graceful degradation |

#### 🔧 구현 특이사항
1. **설정 우선순위**: 환경 변수 > YAML 파일 > 기본값
2. **Graceful Degradation**: Langfuse 미설치/비활성화 시 에러 없이 동작
3. **Correlation ID**: contextvars 기반으로 비동기 환경에서 안전하게 추적
4. **예외 계층**: `AgentOrchestratorError` 기본 클래스에서 상속
5. **표준 에러 응답**: `create_error_response()` 함수로 일관된 형식 제공

#### 📦 새로 추가된 파일
| 파일 | 설명 |
|------|------|
| `src/utils/config.py` | 설정 관리 (AppConfig, 서브 설정 클래스) |
| `src/utils/logging.py` | 구조화 로깅 (structlog, LoggerAdapter) |
| `src/utils/exceptions.py` | 커스텀 예외 클래스 계층 |
| `src/utils/error_handlers.py` | FastAPI 글로벌 에러 핸들러 |
| `src/utils/observability.py` | Langfuse 클라이언트 래퍼 |
| `configs/app.yaml` | 기본 애플리케이션 설정 |
| `tests/unit/test_utils.py` | 유틸리티 모듈 테스트 (56개) |

#### ⚠️ 주의사항
1. **Langfuse 선택적 연동**: MessageBus/BaseAgent 자동 연동은 Phase 8에서 구현 권장
2. **테스트 총합**: Phase 7 단위 테스트 56개 전체 통과 (총 278개)
3. **ruff/mypy 통과**: 코드 품질 검사 통과
4. **langfuse 패키지**: `requirements.txt`에 `langfuse>=2.0.0` 추가됨

---

## Phase 8: 애플리케이션 진입점

### 8.1 Main 모듈 (`src/main.py`)
- [x] FastAPI 앱 인스턴스 생성
- [x] 의존성 주입 설정
- [x] 라우터 등록
- [x] 미들웨어 설정 (CORS, 로깅)
- [x] 시작 시 Agent 로딩
- [x] Graceful shutdown 처리

### 8.2 실행 스크립트
- [x] `uvicorn` 실행 설정
- [x] 개발 모드 설정 (hot-reload)
- [x] 프로덕션 모드 설정

### 📋 Phase 8 이슈 및 참고사항

#### ✅ 구현된 컴포넌트
| 컴포넌트 | 파일 | 테스트 수 | 설명 |
|---------|------|----------|------|
| Main Application | `src/main.py` | 19 | FastAPI 앱, lifespan, 미들웨어 |
| Server Runner | `scripts/run_server.py` | - | CLI 기반 서버 실행 스크립트 |

#### 🔧 구현 특이사항
1. **Lifespan Context Manager**: `@asynccontextmanager`를 사용한 startup/shutdown 이벤트 처리
2. **자동 Agent 로딩**: `configs/agents/` 디렉토리의 YAML 파일에서 Agent 자동 로드
3. **Request ID 미들웨어**: 모든 요청에 `X-Request-ID` 헤더 자동 추가/전파
4. **CORS 설정**: Development 환경에서는 모든 origin 허용
5. **Health Probes**: `/live` (liveness), `/ready` (readiness) 엔드포인트 제공
6. **로깅 미들웨어**: 모든 요청/응답 로깅 (method, path, status_code)

#### 📦 새로 추가된 파일
| 파일 | 설명 |
|------|------|
| `src/main.py` | FastAPI 앱 생성, 의존성 초기화, 미들웨어, lifespan 관리 |
| `scripts/run_server.py` | CLI 스크립트 (--mode dev/prod, --host, --port, --workers) |
| `tests/unit/test_main.py` | Main 모듈 단위 테스트 (19개) |

#### 🚀 서버 실행 방법
```bash
# 개발 모드 (hot-reload)
python scripts/run_server.py --mode dev

# 프로덕션 모드 (multi-worker)
python scripts/run_server.py --mode prod --workers 4

# 커스텀 포트
python scripts/run_server.py --port 8080

# 직접 uvicorn 실행
uvicorn src.main:app --reload --port 8000
```

#### ⚠️ 참고사항
1. **API 문서**: Development 모드에서 `/docs` (Swagger), `/redoc` 자동 활성화
2. **테스트 총합**: Phase 8 단위 테스트 19개 전체 통과 (총 297개)
3. **ruff/mypy 통과**: 코드 품질 검사 통과

---

## Phase 9: 예제 및 문서

### 9.1 예제 작성
- [x] `examples/simple_pipeline.py` - 기본 사용 예제
- [x] `examples/custom_agent.py` - 커스텀 Agent 작성 예제
- [x] `examples/parallel_research.py` - 병렬 처리 예제

### 9.2 문서화
- [x] `README.md` 완성
  - [x] 프로젝트 소개
  - [x] 빠른 시작 가이드
  - [x] 설치 방법
  - [x] 기본 사용법
- [x] API 문서 (자동 생성 Swagger/OpenAPI 확인)
- [x] Agent 작성 가이드 (`docs/AGENT_GUIDE.md`)

### 📋 Phase 9 이슈 및 참고사항

#### ✅ 구현된 컴포넌트
| 파일 | 설명 |
|------|------|
| `examples/simple_pipeline.py` | 순차 파이프라인 예제 (Research → Code → Review) |
| `examples/custom_agent.py` | 커스텀 Agent 작성 예제 (번역, 요약, 감정분석 Agent) |
| `examples/parallel_research.py` | 병렬 검색 예제 (뉴스, 학술, 소셜 미디어 동시 검색) |
| `docs/AGENT_GUIDE.md` | Agent 작성 가이드 문서 |

#### 🔧 예제 실행 방법
```bash
# 기본 파이프라인 예제
python examples/simple_pipeline.py

# 커스텀 Agent 예제
python examples/custom_agent.py

# 병렬 처리 예제
python examples/parallel_research.py
```

#### 📖 API 문서 접근
서버 실행 후:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

> 참고: API 문서는 Development 모드에서만 활성화됩니다.

#### 🔧 예제 특이사항
1. **simple_pipeline.py**: 대화형 메뉴로 예제 선택 가능
2. **custom_agent.py**: LLM 없이 동작하는 감정분석 Agent 포함
3. **parallel_research.py**: 시뮬레이션 기반 (실제 API 호출 없음)
4. **pyproject.toml**: `examples/` 디렉토리에 대한 ruff 예외 설정 추가

#### ⚠️ 참고사항
1. LLM이 필요한 예제는 `ANTHROPIC_API_KEY` 환경 변수가 필요합니다.
2. 감정분석 Agent (custom_agent.py)는 LLM 없이도 동작합니다.
3. 병렬 예제는 시뮬레이션 데이터를 사용합니다.

---

## Phase 10: 테스트 및 품질

### 10.1 테스트 완성
- [x] 단위 테스트 커버리지 80% 이상 (84% 달성)
- [x] 통합 테스트 시나리오 작성 (24개 API 통합 테스트)
- [ ] E2E 테스트 (선택)

### 10.2 코드 품질
- [x] `black` 포매팅 적용
- [x] `ruff` 린팅 통과
- [x] `mypy` 타입 체크 통과

### 10.3 CI/CD (선택)
- [ ] GitHub Actions 워크플로우 작성
- [ ] 자동 테스트 실행
- [ ] 린팅 체크

### 📋 Phase 10 이슈 및 참고사항

#### ✅ 테스트 및 품질 현황
| 항목 | 결과 | 상세 |
|------|------|------|
| 전체 테스트 | 321개 통과 | 단위 297개 + 통합 24개 |
| 코드 커버리지 | 84% | 목표 80% 달성 |
| black 포매팅 | ✅ 통과 | 48개 파일 포매팅 완료 |
| ruff 린팅 | ✅ 통과 | 모든 이슈 해결 |
| mypy 타입 체크 | ✅ 통과 | 34개 소스 파일 검증 |

#### 📊 모듈별 커버리지
| 모듈 | 커버리지 | 비고 |
|------|----------|------|
| src/api/schemas.py | 100% | 완벽한 커버리지 |
| src/models/agent.py | 100% | 완벽한 커버리지 |
| src/models/conversation.py | 97% | |
| src/agents/implementations/*.py | 91-97% | |
| src/core/*.py | 82-94% | |
| src/patterns/*.py | 61-92% | debate 패턴 일부 미커버 |
| src/utils/*.py | 36-97% | observability 연동 코드 미사용 |

#### 🔧 품질 검사 명령어
```bash
# 전체 테스트 (커버리지 포함)
pytest tests/ -v --cov=src --cov-report=term-missing

# HTML 커버리지 리포트 생성
pytest tests/ --cov=src --cov-report=html

# 코드 품질 검사
black src/ tests/ --check
ruff check src/ tests/
mypy src/ --ignore-missing-imports
```

#### ⚠️ 참고사항
1. **커버리지 미달 영역**: `src/utils/observability.py` (36%) - Langfuse 연동 코드가 실제 환경에서만 동작
2. **Debate 패턴**: 일부 복잡한 시나리오 (61%) - 모더레이터 로직 등
3. **E2E 테스트**: 선택 사항으로 남김 (실제 LLM 호출 필요)
4. **CI/CD**: GitHub Actions 설정은 선택 사항으로 남김

---

## 추가 기능 (선택적)

### WebSocket 지원
- [ ] WebSocket 엔드포인트 구현
- [ ] 실시간 진행 상황 스트리밍
- [ ] 연결 관리

### Agent Hot-reload
- [ ] 파일 변경 감지 (watchdog)
- [ ] 런타임 Agent 재로딩
- [ ] 무중단 업데이트

### 대시보드
- [ ] 간단한 웹 UI
- [ ] Agent 상태 표시
- [ ] 대화 로그 뷰어

### 캐싱
- [ ] Redis 캐싱 레이어
- [ ] 결과 캐싱
- [ ] 중복 요청 방지

---

## 진행 상황 요약

| Phase | 항목 | 완료 |
|-------|------|------|
| Phase 1 | 프로젝트 초기 설정 | ✅ (1.1, 1.2 완료) |
| Phase 2 | 데이터 모델 | ✅ (36개 테스트 통과) |
| Phase 3 | 핵심 컴포넌트 | ✅ (63개 테스트 통과) |
| Phase 4 | Agent 구현 | ✅ (80개 테스트 통과) |
| Phase 5 | 대화 패턴 | ✅ (43개 테스트 통과) |
| Phase 6 | API 구현 | ✅ (24개 테스트 통과) |
| Phase 7 | 유틸리티 | ✅ (56개 테스트 통과) |
| Phase 8 | 앱 진입점 | ✅ (19개 테스트 통과) |
| Phase 9 | 예제/문서 | ✅ (3개 예제, Agent 가이드 완료) |
| Phase 10 | 테스트/품질 | ✅ (321개 테스트, 84% 커버리지) |

---

## 참고 명령어

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt

# 개발 서버 실행
uvicorn src.main:app --reload --port 8000

# 테스트 실행
pytest tests/ -v

# 포매팅
black src/ tests/

# 린팅
ruff check src/ tests/

# 타입 체크
mypy src/
```

---

*마지막 업데이트: 2025-12-01*
