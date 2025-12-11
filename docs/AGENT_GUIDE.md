# Agent 작성 가이드

> Agent Orchestrator에서 커스텀 Agent를 작성하는 방법

---

## 목차

1. [개요](#개요)
2. [YAML 설정으로 Agent 추가](#yaml-설정으로-agent-추가)
3. [코드로 커스텀 Agent 작성](#코드로-커스텀-agent-작성)
4. [Agent 라이프사이클](#agent-라이프사이클)
5. [LLM 호출](#llm-호출)
6. [에러 처리](#에러-처리)
7. [테스트](#테스트)
8. [예제](#예제)

---

## 개요

Agent Orchestrator에서 Agent를 추가하는 방법은 두 가지입니다:

1. **YAML 설정 파일** - 가장 간단한 방법. `configs/agents/` 디렉토리에 YAML 파일만 추가
2. **Python 코드** - `BaseAgent`를 상속하여 커스텀 로직 구현

---

## YAML 설정으로 Agent 추가

### 기본 구조

`configs/agents/` 디렉토리에 `.yaml` 파일을 생성하면 서버 시작 시 자동으로 로드됩니다.

```yaml
# configs/agents/my_agent.yaml

# 필수 필드
agent_id: "my_agent_001"       # 고유 식별자
name: "My Agent"               # 표시 이름
description: "설명"            # Agent 설명

# 능력(Capabilities) 정의
capabilities:
  - name: "my_capability"      # 능력 이름 (다른 Agent와 구분)
    description: "능력 설명"
    input_schema:              # 입력 JSON 스키마
      type: "object"
      properties:
        query:
          type: "string"
          description: "입력 쿼리"
      required: ["query"]
    output_schema:             # 출력 JSON 스키마
      type: "object"
      properties:
        result:
          type: "string"

# 선택적 필드
model: "claude-haiku-4-5-20251001"  # 사용할 LLM 모델
max_tokens: 4096                     # 최대 토큰 수
temperature: 0.7                     # 생성 온도 (0.0 ~ 2.0)

# 시스템 프롬프트 (Agent의 역할과 지침)
system_prompt: |
  당신은 전문 분석가입니다.
  주어진 데이터를 분석하고 인사이트를 제공하세요.
  응답은 항상 구조화된 형식으로 제공하세요.

# 메타데이터 (선택적)
metadata:
  version: "1.0"
  author: "Your Name"
```

### 전체 예제

```yaml
# configs/agents/data_analyst.yaml
agent_id: "data_analyst_001"
name: "Data Analyst Agent"
description: "데이터 분석 및 인사이트 도출 전문 Agent"

capabilities:
  - name: "data_analysis"
    description: "데이터셋 분석 및 통계 요약"
    input_schema:
      type: "object"
      properties:
        data:
          type: "string"
          description: "분석할 데이터 (CSV/JSON)"
        analysis_type:
          type: "string"
          enum: ["summary", "correlation", "trend"]
          default: "summary"
      required: ["data"]
    output_schema:
      type: "object"
      properties:
        summary:
          type: "object"
        insights:
          type: "array"
          items:
            type: "string"

  - name: "visualization_suggestion"
    description: "데이터에 적합한 시각화 방법 제안"
    input_schema:
      type: "object"
      properties:
        data_description:
          type: "string"
    output_schema:
      type: "object"
      properties:
        suggested_charts:
          type: "array"

model: "claude-haiku-4-5-20251001"
max_tokens: 8192
temperature: 0.3

system_prompt: |
  당신은 전문 데이터 분석가입니다.

  역할:
  - 데이터를 분석하고 핵심 인사이트를 도출합니다
  - 통계적 분석 결과를 이해하기 쉽게 설명합니다
  - 적절한 시각화 방법을 제안합니다

  응답 형식:
  - 항상 JSON 형식으로 응답하세요
  - 인사이트는 bullet point로 정리하세요
  - 수치는 소수점 2자리까지 표시하세요

metadata:
  version: "1.0"
  domain: "data-science"
```

---

## 코드로 커스텀 Agent 작성

### 기본 구조

`BaseAgent`를 상속하여 `process()` 메서드를 구현합니다.

```python
from src.agents.base import BaseAgent
from src.models import AgentConfig, Message

class MyCustomAgent(BaseAgent):
    """커스텀 Agent 구현."""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        # 추가 초기화 로직

    async def process(self, message: Message) -> Message:
        """메시지를 처리하고 응답을 반환합니다."""
        # 상태를 'busy'로 설정
        self.set_busy()

        try:
            # 메시지에서 데이터 추출
            content = message.content
            query = content.get("query") or content.get("task", {}).get("query", "")

            # 작업 수행 (LLM 호출 또는 커스텀 로직)
            result = await self._call_llm(f"처리해주세요: {query}")

            # 상태를 'active'로 복원
            self.activate()

            # 응답 생성
            return self._create_response(
                message=message,
                content={"result": result}
            )

        except Exception as e:
            # 에러 상태 설정
            self.set_error(str(e))
            return self._create_error_response(message, str(e))
```

### YAML과 연결하기

커스텀 Agent 클래스를 YAML 설정과 연결하려면 `agent_class` 필드를 사용합니다.

```yaml
# configs/agents/my_custom.yaml
agent_id: "my_custom_001"
name: "My Custom Agent"
agent_class: "src.agents.implementations.my_module.MyCustomAgent"
capabilities:
  - name: "custom_task"
    description: "커스텀 작업"
```

### BaseAgent 주요 메서드

| 메서드 | 설명 |
|--------|------|
| `process(message)` | **필수 구현**. 메시지 처리 및 응답 반환 |
| `can_handle(capability)` | 특정 capability 처리 가능 여부 확인 |
| `get_capabilities()` | 지원하는 capability 목록 반환 |
| `health_check()` | Agent 상태 확인 |
| `_call_llm(prompt, context)` | LLM 호출 헬퍼 |
| `_create_response(message, content)` | 응답 메시지 생성 헬퍼 |
| `_create_error_response(message, error)` | 에러 응답 생성 헬퍼 |

### 상태 관리 메서드

| 메서드 | 설명 |
|--------|------|
| `activate()` | 상태를 ACTIVE로 설정 |
| `deactivate()` | 상태를 INACTIVE로 설정 |
| `set_busy()` | 상태를 BUSY로 설정 |
| `set_error(message)` | 상태를 ERROR로 설정 |

---

## Agent 라이프사이클

```
[생성] → [등록] → [활성화] → [작업 처리] → [비활성화/오류]
   ↓        ↓         ↓           ↓              ↓
AgentConfig  Registry  ACTIVE    BUSY → ACTIVE   INACTIVE/ERROR
```

### 상태 다이어그램

```
STARTING → ACTIVE ⇄ BUSY
              ↓       ↓
          INACTIVE  ERROR
              ↓
          STOPPING
```

### 상태별 설명

| 상태 | 설명 |
|------|------|
| `STARTING` | Agent 초기화 중 |
| `ACTIVE` | 정상 작동, 작업 수신 가능 |
| `BUSY` | 작업 처리 중 |
| `INACTIVE` | 비활성화됨 |
| `ERROR` | 오류 발생 |
| `STOPPING` | 종료 중 |

---

## LLM 호출

### 기본 사용법

```python
async def process(self, message: Message) -> Message:
    # 간단한 LLM 호출
    result = await self._call_llm("질문에 답해주세요: ...")

    # 컨텍스트와 함께 호출
    result = await self._call_llm(
        prompt="질문: ...",
        context={"previous_result": "이전 결과"}
    )
```

### 커스텀 시스템 프롬프트

```python
# config에서 설정
config = AgentConfig(
    agent_id="my_agent",
    name="My Agent",
    system_prompt="당신은 전문가입니다..."
)

# 또는 _call_llm에서 직접 전달
result = await self._call_llm(
    prompt="...",
    system_prompt="이 작업에 특화된 프롬프트..."
)
```

### LLM 설정 옵션

```python
config = AgentConfig(
    agent_id="my_agent",
    name="My Agent",
    model="claude-haiku-4-5-20251001",  # 모델 선택
    max_tokens=4096,                      # 최대 출력 토큰
    temperature=0.7,                       # 생성 다양성 (0.0~2.0)
)
```

---

## 에러 처리

### 권장 패턴

```python
async def process(self, message: Message) -> Message:
    self.set_busy()

    try:
        # 작업 수행
        result = await self._do_work(message)
        self.activate()
        return self._create_response(message, {"result": result})

    except ValidationError as e:
        # 입력 검증 오류
        self.activate()  # 복구 가능한 오류
        return self._create_error_response(message, f"입력 오류: {e}")

    except LLMError as e:
        # LLM 호출 오류
        self.set_error(str(e))
        return self._create_error_response(message, f"LLM 오류: {e}")

    except Exception as e:
        # 예상치 못한 오류
        self.set_error(str(e))
        return self._create_error_response(message, f"내부 오류: {e}")
```

### 커스텀 예외

```python
from src.utils.exceptions import AgentOrchestratorError

class MyAgentError(AgentOrchestratorError):
    """커스텀 Agent 오류."""
    pass
```

---

## 테스트

### 단위 테스트 예제

```python
# tests/unit/test_my_agent.py
import pytest
from unittest.mock import AsyncMock, patch

from src.agents.implementations.my_agent import MyCustomAgent
from src.models import AgentConfig, AgentCapability, Message


@pytest.fixture
def agent_config():
    return AgentConfig(
        agent_id="test_agent",
        name="Test Agent",
        capabilities=[
            AgentCapability(name="test_cap", description="Test")
        ],
    )


@pytest.fixture
def agent(agent_config):
    return MyCustomAgent(agent_config)


class TestMyCustomAgent:
    """MyCustomAgent 테스트."""

    def test_init(self, agent):
        """초기화 테스트."""
        assert agent.config.agent_id == "test_agent"
        assert agent.is_active

    def test_can_handle(self, agent):
        """capability 처리 가능 여부 테스트."""
        assert agent.can_handle("test_cap")
        assert not agent.can_handle("unknown")

    @pytest.mark.asyncio
    async def test_process_success(self, agent):
        """성공적인 처리 테스트."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="test_agent",
            content={"query": "테스트"},
            correlation_id="test_conv",
        )

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "LLM 응답"

            response = await agent.process(message)

            assert response.message_type.value == "response"
            assert "result" in response.content

    @pytest.mark.asyncio
    async def test_process_error(self, agent):
        """에러 처리 테스트."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="test_agent",
            content={},  # 빈 컨텐츠
            correlation_id="test_conv",
        )

        with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM 오류")

            response = await agent.process(message)

            assert response.message_type.value == "error"
```

### 테스트 실행

```bash
# 특정 Agent 테스트
pytest tests/unit/test_my_agent.py -v

# 모든 Agent 테스트
pytest tests/unit/test_agent*.py -v
```

---

## 예제

### 1. 번역 Agent

```python
class TranslatorAgent(BaseAgent):
    """텍스트 번역 Agent."""

    async def process(self, message: Message) -> Message:
        self.set_busy()

        try:
            text = message.content.get("text", "")
            target_lang = message.content.get("target_lang", "한국어")

            prompt = f"다음을 {target_lang}로 번역하세요:\n{text}"
            result = await self._call_llm(prompt)

            self.activate()
            return self._create_response(message, {
                "result": result,
                "translated_text": result,
                "target_lang": target_lang,
            })

        except Exception as e:
            self.set_error(str(e))
            return self._create_error_response(message, str(e))
```

### 2. 데이터 검증 Agent (LLM 없이)

```python
class DataValidatorAgent(BaseAgent):
    """데이터 검증 Agent (규칙 기반)."""

    async def process(self, message: Message) -> Message:
        self.set_busy()

        try:
            data = message.content.get("data", {})
            rules = message.content.get("rules", [])

            errors = []
            for rule in rules:
                if not self._check_rule(data, rule):
                    errors.append(f"규칙 위반: {rule}")

            self.activate()
            return self._create_response(message, {
                "result": {
                    "valid": len(errors) == 0,
                    "errors": errors,
                    "checked_rules": len(rules),
                }
            })

        except Exception as e:
            self.set_error(str(e))
            return self._create_error_response(message, str(e))

    def _check_rule(self, data: dict, rule: str) -> bool:
        # 규칙 검증 로직
        return True
```

### 3. 외부 API 연동 Agent

```python
import httpx

class WeatherAgent(BaseAgent):
    """날씨 정보 조회 Agent."""

    API_URL = "https://api.weather.example.com"

    async def process(self, message: Message) -> Message:
        self.set_busy()

        try:
            city = message.content.get("city", "Seoul")

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.API_URL}/weather",
                    params={"city": city}
                )
                weather_data = response.json()

            self.activate()
            return self._create_response(message, {
                "result": weather_data,
                "city": city,
            })

        except httpx.HTTPError as e:
            self.set_error(str(e))
            return self._create_error_response(message, f"API 오류: {e}")
```

---

## 참고 자료

- [PLANNING.md](./PLANNING.md) - 전체 시스템 설계
- [examples/custom_agent.py](../examples/custom_agent.py) - 커스텀 Agent 예제
- [src/agents/base.py](../src/agents/base.py) - BaseAgent 구현

---

*마지막 업데이트: 2025-12-01*
