# API 예시 - 대화 패턴별 Task 생성

> 서버 실행: `python scripts/run_server.py --mode dev`
>
> Swagger UI: http://localhost:8000/docs

---

## 1. Sequential (순차) 패턴

Research → Code → Review 순서로 실행됩니다.

```bash
curl -X POST "http://localhost:8000/api/v1/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "웹 스크래퍼 개발",
    "pattern": "sequential",
    "input": {
      "topic": "Python으로 뉴스 웹사이트 스크래퍼 만들기"
    },
    "stages": [
      {"name": "research", "agent_capability": "web_search"},
      {"name": "code", "agent_capability": "code_generation"},
      {"name": "review", "agent_capability": "code_review"}
    ],
    "timeout_seconds": 300
  }'
```

**Windows CMD:**
```cmd
curl -X POST "http://localhost:8000/api/v1/tasks" -H "Content-Type: application/json" -d "{\"name\": \"웹 스크래퍼 개발\", \"pattern\": \"sequential\", \"input\": {\"topic\": \"Python으로 뉴스 웹사이트 스크래퍼 만들기\"}, \"stages\": [{\"name\": \"research\", \"agent_capability\": \"web_search\"}, {\"name\": \"code\", \"agent_capability\": \"code_generation\"}, {\"name\": \"review\", \"agent_capability\": \"code_review\"}], \"timeout_seconds\": 300}"
```

---

## 2. Parallel (병렬) 패턴

여러 Agent가 동시에 실행되고 결과가 병합됩니다.

```bash
curl -X POST "http://localhost:8000/api/v1/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "종합 코드 분석",
    "pattern": "parallel",
    "input": {
      "code": "def calculate(x, y): return x/y"
    },
    "stages": [
      {"name": "security", "agent_capability": "security_review"},
      {"name": "performance", "agent_capability": "performance_review"},
      {"name": "style", "agent_capability": "style_review"}
    ],
    "timeout_seconds": 300
  }'
```

**Windows CMD:**
```cmd
curl -X POST "http://localhost:8000/api/v1/tasks" -H "Content-Type: application/json" -d "{\"name\": \"종합 코드 분석\", \"pattern\": \"parallel\", \"input\": {\"code\": \"def calculate(x, y): return x/y\"}, \"stages\": [{\"name\": \"security\", \"agent_capability\": \"security_review\"}, {\"name\": \"performance\", \"agent_capability\": \"performance_review\"}, {\"name\": \"style\", \"agent_capability\": \"style_review\"}], \"timeout_seconds\": 300}"
```

---

## 3. Debate (토론) 패턴

두 Agent가 서로 의견을 교환하며 토론합니다.

```bash
curl -X POST "http://localhost:8000/api/v1/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "AI 일자리 대체 토론",
    "pattern": "debate",
    "input": {
      "topic": "AI가 인간의 일자리를 대체하는 것에 대해 토론해주세요"
    },
    "stages": [
      {"name": "optimist_view", "agent_capability": "debate"},
      {"name": "pessimist_view", "agent_capability": "debate"}
    ],
    "timeout_seconds": 300,
    "max_iterations": 3
  }'
```

**Windows CMD:**
```cmd
curl -X POST "http://localhost:8000/api/v1/tasks" -H "Content-Type: application/json" -d "{\"name\": \"AI 일자리 대체 토론\", \"pattern\": \"debate\", \"input\": {\"topic\": \"AI가 인간의 일자리를 대체하는 것에 대해 토론해주세요\"}, \"stages\": [{\"name\": \"optimist_view\", \"agent_capability\": \"debate\"}, {\"name\": \"pessimist_view\", \"agent_capability\": \"debate\"}], \"timeout_seconds\": 300, \"max_iterations\": 3}"
```

---

## 4. Router (라우터) 패턴

입력에 따라 적절한 Agent로 라우팅됩니다.

```bash
curl -X POST "http://localhost:8000/api/v1/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "코드 질문 처리",
    "pattern": "router",
    "input": {
      "question": "이 코드에서 버그를 찾아줘: for i in range(10): print(i"
    },
    "stages": [
      {"name": "debug", "agent_capability": "debugging"},
      {"name": "explain", "agent_capability": "code_explanation"},
      {"name": "generate", "agent_capability": "code_generation"}
    ],
    "timeout_seconds": 300
  }'
```

**Windows CMD:**
```cmd
curl -X POST "http://localhost:8000/api/v1/tasks" -H "Content-Type: application/json" -d "{\"name\": \"코드 질문 처리\", \"pattern\": \"router\", \"input\": {\"question\": \"이 코드에서 버그를 찾아줘: for i in range(10): print(i\"}, \"stages\": [{\"name\": \"debug\", \"agent_capability\": \"debugging\"}, {\"name\": \"explain\", \"agent_capability\": \"code_explanation\"}, {\"name\": \"generate\", \"agent_capability\": \"code_generation\"}], \"timeout_seconds\": 300}"
```

---

## 5. Hierarchical (계층) 패턴

Supervisor가 Worker들에게 작업을 분배합니다.

```bash
curl -X POST "http://localhost:8000/api/v1/tasks" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "프로젝트 분석",
    "pattern": "hierarchical",
    "input": {
      "project": "전자상거래 웹사이트 개발"
    },
    "stages": [
      {"name": "supervisor", "agent_capability": "web_search"},
      {"name": "worker_code", "agent_capability": "code_generation"},
      {"name": "worker_review", "agent_capability": "code_review"}
    ],
    "timeout_seconds": 300
  }'
```

**Windows CMD:**
```cmd
curl -X POST "http://localhost:8000/api/v1/tasks" -H "Content-Type: application/json" -d "{\"name\": \"프로젝트 분석\", \"pattern\": \"hierarchical\", \"input\": {\"project\": \"전자상거래 웹사이트 개발\"}, \"stages\": [{\"name\": \"supervisor\", \"agent_capability\": \"web_search\"}, {\"name\": \"worker_code\", \"agent_capability\": \"code_generation\"}, {\"name\": \"worker_review\", \"agent_capability\": \"code_review\"}], \"timeout_seconds\": 300}"
```

---

## 결과 확인 방법

### Task 상태 확인
```bash
curl http://localhost:8000/api/v1/tasks/{task_id}
```

### Task 결과 확인
```bash
curl http://localhost:8000/api/v1/tasks/{task_id}/result
```

### 대화 목록 확인
```bash
curl http://localhost:8000/api/v1/conversations
```

### 대화 메시지 확인
```bash
curl http://localhost:8000/api/v1/conversations/{conversation_id}/messages
```

---

## 등록된 Agent 확인

```bash
curl http://localhost:8000/api/v1/agents
```

### 기본 Agent Capabilities

| Agent | Capabilities |
|-------|-------------|
| researcher_001 | web_search, summarization, data_extraction |
| coder_001 | code_generation, code_modification, code_explanation, debugging |
| reviewer_001 | code_review, security_review, performance_review, style_review |
| optimist | debate |
| pessimist | debate |
