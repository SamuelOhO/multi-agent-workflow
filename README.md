# Agent Orchestrator

> Multi-Agent 협업 시스템: Agent를 쉽게 추가하고, 자동으로 다른 Agent들과 대화/협업할 수 있는 프레임워크

## 서버 실행 방법

```bash
venv\Scripts\activate
# 개발 모드 (hot-reload)
python scripts/run_server.py --mode dev

# 프로덕션 모드 (multi-worker)
python scripts/run_server.py --mode prod --workers 4

# 직접 uvicorn 실행
uvicorn src.main:app --reload --port 8000
```

---

## 개요

Agent Orchestrator는 여러 AI Agent가 서로 대화하고 협업하여 복잡한 작업을 수행하는 시스템입니다. 새로운 Agent를 추가하면 별도 설정 없이 자동으로 기존 시스템과 통합되어 협업이 가능합니다.

## 핵심 기능

- **표준화된 Agent 인터페이스**: 모든 Agent가 동일한 방식으로 통신
- **자동 발견(Auto-Discovery)**: 새 Agent 추가 시 자동 등록 및 통합
- **유연한 대화 패턴**: 순차, 병렬, 토론 등 다양한 협업 방식 지원
- **확장성**: Agent 수가 늘어나도 시스템 성능 유지

## 시스템 요구사항

- Python 3.11+
- Anthropic API Key

## 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/your-org/agent-orchestrator.git
cd agent-orchestrator

# 가상환경 생성 및 활성화
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 설정

```bash
# 환경 변수 파일 생성
cp .env.example .env

# .env 파일을 열어 API 키 설정
# ANTHROPIC_API_KEY=your_api_key_here
```

### 3. 서버 실행

```bash
# 개발 모드 (hot-reload)
uvicorn src.main:app --reload --port 8000

# 프로덕션 모드
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 4. API 문서 확인

서버 실행 후 브라우저에서:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 프로젝트 구조

```
agent-orchestrator/
├── src/
│   ├── core/           # 핵심 로직 (Orchestrator, Registry, MessageBus)
│   ├── agents/         # Agent 구현
│   ├── patterns/       # 대화 패턴 (Sequential, Parallel, Debate)
│   ├── api/            # REST API
│   ├── models/         # 데이터 모델
│   └── utils/          # 유틸리티
├── configs/
│   ├── agents/         # Agent 설정 (YAML)
│   └── conversations/  # 대화 패턴 설정
├── tests/              # 테스트
├── examples/           # 사용 예제
└── docs/               # 문서
```

## 사용 예시

### 기본 작업 실행

```python
from src.core.orchestrator import Orchestrator

orchestrator = Orchestrator()

# 작업 실행
result = await orchestrator.execute(
    task="웹 스크래퍼를 만들어줘",
    pattern="sequential"
)
```

### 커스텀 Agent 추가

`configs/agents/` 디렉토리에 YAML 파일을 추가하면 자동으로 시스템에 통합됩니다:

```yaml
# configs/agents/my_agent.yaml
agent_id: "my_agent_001"
name: "My Custom Agent"
description: "커스텀 작업을 수행하는 Agent"

capabilities:
  - name: "custom_task"
    description: "커스텀 작업 수행"
    input_schema:
      type: "object"
      properties:
        input:
          type: "string"
    output_schema:
      type: "object"
      properties:
        result:
          type: "string"

model: "claude-haiku-4-5-20251001"
max_tokens: 4096
temperature: 0.7
```

## 지원하는 대화 패턴

| 패턴 | 설명 | 사용 사례 |
|------|------|----------|
| Sequential | 순차적 Agent 실행 | 파이프라인 처리 |
| Parallel | 병렬 실행 후 결과 병합 | 다중 소스 정보 수집 |
| Debate | 토론 후 합의 도출 | 의사결정, 코드 리뷰 |
| Router | 조건별 Agent 라우팅 | 질문 유형별 처리 |

## 개발

### 테스트 실행

```bash
# 전체 테스트
pytest tests/ -v

# 커버리지 포함
pytest tests/ --cov=src --cov-report=html
```

### 코드 품질

```bash
# 포매팅
black src/ tests/

# 린팅
ruff check src/ tests/

# 타입 체크
mypy src/
```

## 문서

자세한 내용은 [docs/PLANNING.md](docs/PLANNING.md)를 참조하세요.

## 라이선스

MIT License

## 기여

기여를 환영합니다! Issue와 Pull Request를 자유롭게 제출해주세요.
