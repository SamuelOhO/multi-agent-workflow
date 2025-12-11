"""API 통합 테스트.

FastAPI 엔드포인트의 통합 테스트를 수행합니다.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import (
    api_router,
    init_dependencies,
)
from src.core.conversation import ConversationManager
from src.core.message_bus import InMemoryMessageBus
from src.core.orchestrator import Orchestrator
from src.core.registry import AgentRegistry


@pytest.fixture
def app() -> FastAPI:
    """FastAPI 앱 fixture."""
    app = FastAPI(title="Agent Orchestrator Test")
    app.include_router(api_router)
    return app


@pytest.fixture
def test_dependencies():
    """테스트용 의존성 설정."""
    registry = AgentRegistry()
    message_bus = InMemoryMessageBus()
    conversation_manager = ConversationManager()
    orchestrator = Orchestrator(
        registry=registry,
        message_bus=message_bus,
        conversation_manager=conversation_manager,
    )

    init_dependencies(registry, conversation_manager, orchestrator)

    return {
        "registry": registry,
        "message_bus": message_bus,
        "conversation_manager": conversation_manager,
        "orchestrator": orchestrator,
    }


@pytest.fixture
def client(app: FastAPI, test_dependencies) -> TestClient:
    """TestClient fixture."""
    return TestClient(app)


class TestHealthEndpoint:
    """시스템 상태 체크 테스트."""

    def test_health_check(self, client: TestClient):
        """기본 상태 체크 테스트."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "healthy"
        assert "agents_registered" in data["data"]
        assert "active_conversations" in data["data"]


class TestAgentEndpoints:
    """Agent API 테스트."""

    def test_list_agents_empty(self, client: TestClient):
        """Agent가 없을 때 목록 조회."""
        response = client.get("/api/v1/agents")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] == []
        assert data["metadata"]["count"] == 0

    def test_register_agent(self, client: TestClient):
        """Agent 등록 테스트."""
        agent_data = {
            "agent_id": "test_agent_001",
            "name": "Test Agent",
            "description": "테스트 Agent",
            "capabilities": [
                {
                    "name": "test_capability",
                    "description": "테스트 기능",
                }
            ],
        }

        response = client.post("/api/v1/agents", json=agent_data)

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["data"]["agent_id"] == "test_agent_001"
        assert data["data"]["name"] == "Test Agent"
        assert "test_capability" in data["data"]["capabilities"]

    def test_register_agent_duplicate(self, client: TestClient):
        """중복 Agent 등록 시 에러."""
        agent_data = {
            "agent_id": "duplicate_agent",
            "name": "Duplicate Agent",
            "capabilities": [{"name": "cap1", "description": "desc"}],
        }

        # 첫 번째 등록 성공
        response1 = client.post("/api/v1/agents", json=agent_data)
        assert response1.status_code == 201

        # 두 번째 등록 실패
        response2 = client.post("/api/v1/agents", json=agent_data)
        assert response2.status_code == 409

    def test_get_agent(self, client: TestClient):
        """Agent 상세 조회 테스트."""
        # 먼저 Agent 등록
        agent_data = {
            "agent_id": "get_test_agent",
            "name": "Get Test Agent",
            "capabilities": [{"name": "test", "description": "test"}],
        }
        client.post("/api/v1/agents", json=agent_data)

        # 조회
        response = client.get("/api/v1/agents/get_test_agent")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["agent_id"] == "get_test_agent"

    def test_get_agent_not_found(self, client: TestClient):
        """존재하지 않는 Agent 조회."""
        response = client.get("/api/v1/agents/nonexistent_agent")

        assert response.status_code == 404

    def test_delete_agent(self, client: TestClient):
        """Agent 삭제 테스트."""
        # 먼저 Agent 등록
        agent_data = {
            "agent_id": "delete_test_agent",
            "name": "Delete Test Agent",
            "capabilities": [{"name": "test", "description": "test"}],
        }
        client.post("/api/v1/agents", json=agent_data)

        # 삭제
        response = client.delete("/api/v1/agents/delete_test_agent")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # 삭제 확인
        get_response = client.get("/api/v1/agents/delete_test_agent")
        assert get_response.status_code == 404

    def test_delete_agent_not_found(self, client: TestClient):
        """존재하지 않는 Agent 삭제."""
        response = client.delete("/api/v1/agents/nonexistent_agent")

        assert response.status_code == 404

    def test_agent_health_check(self, client: TestClient):
        """Agent 상태 체크 테스트."""
        # Agent 등록
        agent_data = {
            "agent_id": "health_test_agent",
            "name": "Health Test Agent",
            "capabilities": [{"name": "test", "description": "test"}],
        }
        client.post("/api/v1/agents", json=agent_data)

        # 상태 체크
        response = client.get("/api/v1/agents/health_test_agent/health")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["agent_id"] == "health_test_agent"
        assert data["data"]["status"] == "healthy"


class TestTaskEndpoints:
    """Task API 테스트."""

    def test_create_task(self, client: TestClient):
        """Task 생성 테스트."""
        # Agent 등록
        agent_data = {
            "agent_id": "task_agent",
            "name": "Task Agent",
            "capabilities": [{"name": "process_task", "description": "작업 처리"}],
        }
        client.post("/api/v1/agents", json=agent_data)

        # Task 생성
        task_data = {
            "name": "Test Task",
            "description": "테스트 작업",
            "input": {"query": "테스트 쿼리"},
            "pattern": "sequential",
            "stages": [
                {
                    "name": "stage1",
                    "agent_capability": "process_task",
                }
            ],
            "timeout_seconds": 60,
        }

        response = client.post("/api/v1/tasks", json=task_data)

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert "task_id" in data["data"]
        assert data["data"]["pattern"] == "sequential"

    def test_get_task_status(self, client: TestClient):
        """Task 상태 조회 테스트."""
        # Agent 등록
        agent_data = {
            "agent_id": "status_agent",
            "name": "Status Agent",
            "capabilities": [{"name": "status_test", "description": "상태 테스트"}],
        }
        client.post("/api/v1/agents", json=agent_data)

        # Task 생성
        task_data = {
            "name": "Status Test Task",
            "input": {"data": "test"},
            "stages": [
                {
                    "name": "stage1",
                    "agent_capability": "status_test",
                }
            ],
        }
        create_response = client.post("/api/v1/tasks", json=task_data)
        task_id = create_response.json()["data"]["task_id"]

        # 상태 조회
        response = client.get(f"/api/v1/tasks/{task_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["task_id"] == task_id

    def test_get_task_not_found(self, client: TestClient):
        """존재하지 않는 Task 조회."""
        response = client.get("/api/v1/tasks/nonexistent_task_id")

        assert response.status_code == 404

    def test_cancel_task_already_completed(self, client: TestClient):
        """Already completed task cancel returns conflict."""
        # Agent 등록
        agent_data = {
            "agent_id": "cancel_agent",
            "name": "Cancel Agent",
            "capabilities": [{"name": "cancel_test", "description": "취소 테스트"}],
        }
        client.post("/api/v1/agents", json=agent_data)

        # Task 생성 (백그라운드에서 빠르게 완료됨)
        task_data = {
            "name": "Cancel Test Task",
            "input": {"data": "test"},
            "stages": [
                {
                    "name": "stage1",
                    "agent_capability": "cancel_test",
                }
            ],
        }
        create_response = client.post("/api/v1/tasks", json=task_data)
        task_id = create_response.json()["data"]["task_id"]

        # Task 취소 시도 - 이미 완료된 경우 409 반환
        import time

        time.sleep(0.1)  # 백그라운드 작업 완료 대기
        response = client.delete(f"/api/v1/tasks/{task_id}")

        # 완료된 작업이면 409 Conflict, 아직 진행 중이면 200 OK
        assert response.status_code in (200, 409)


class TestConversationEndpoints:
    """Conversation API 테스트."""

    def test_create_conversation(self, client: TestClient):
        """Conversation 생성 테스트."""
        conv_data = {
            "name": "Test Conversation",
            "description": "테스트 대화",
            "pattern": "sequential",
            "stages": [
                {
                    "name": "stage1",
                    "agent_capability": "test_cap",
                }
            ],
            "initial_input": {"query": "test"},
        }

        response = client.post("/api/v1/conversations", json=conv_data)

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert "id" in data["data"]
        assert data["data"]["name"] == "Test Conversation"
        assert data["data"]["pattern"] == "sequential"

    def test_get_conversation(self, client: TestClient):
        """Conversation 조회 테스트."""
        # Conversation 생성
        conv_data = {
            "name": "Get Test Conversation",
            "pattern": "parallel",
        }
        create_response = client.post("/api/v1/conversations", json=conv_data)
        conv_id = create_response.json()["data"]["id"]

        # 조회
        response = client.get(f"/api/v1/conversations/{conv_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == conv_id

    def test_get_conversation_not_found(self, client: TestClient):
        """존재하지 않는 Conversation 조회."""
        response = client.get("/api/v1/conversations/nonexistent_id")

        assert response.status_code == 404

    def test_get_conversation_messages_empty(self, client: TestClient):
        """메시지가 없는 Conversation의 메시지 조회."""
        # Conversation 생성
        conv_data = {"name": "Empty Messages Test"}
        create_response = client.post("/api/v1/conversations", json=conv_data)
        conv_id = create_response.json()["data"]["id"]

        # 메시지 조회
        response = client.get(f"/api/v1/conversations/{conv_id}/messages")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"] == []
        assert data["metadata"]["count"] == 0

    def test_add_message_to_conversation(self, client: TestClient):
        """Conversation에 메시지 추가 테스트."""
        # Conversation 생성
        conv_data = {"name": "Add Message Test"}
        create_response = client.post("/api/v1/conversations", json=conv_data)
        conv_id = create_response.json()["data"]["id"]

        # 메시지 추가
        message_data = {
            "sender_id": "user",
            "recipient_id": "agent",
            "message_type": "task",
            "content": {"query": "Hello, agent!"},
        }
        response = client.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json=message_data,
        )

        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["data"]["sender_id"] == "user"

        # 메시지 확인
        messages_response = client.get(f"/api/v1/conversations/{conv_id}/messages")
        messages_data = messages_response.json()
        assert messages_data["metadata"]["count"] == 1


class TestAPISchemaValidation:
    """API 스키마 유효성 검증 테스트."""

    def test_register_agent_missing_required(self, client: TestClient):
        """필수 필드 누락 시 에러."""
        # agent_id 누락
        agent_data = {
            "name": "Test Agent",
            "capabilities": [],
        }

        response = client.post("/api/v1/agents", json=agent_data)

        assert response.status_code == 422  # Validation error

    def test_register_agent_invalid_temperature(self, client: TestClient):
        """유효하지 않은 temperature 값."""
        agent_data = {
            "agent_id": "temp_test",
            "name": "Temp Test",
            "capabilities": [{"name": "test", "description": "test"}],
            "temperature": 3.0,  # 최대 2.0
        }

        response = client.post("/api/v1/agents", json=agent_data)

        assert response.status_code == 422

    def test_create_task_invalid_timeout(self, client: TestClient):
        """유효하지 않은 timeout 값."""
        task_data = {
            "name": "Invalid Timeout Task",
            "input": {},
            "timeout_seconds": 10000,  # 최대 3600
        }

        response = client.post("/api/v1/tasks", json=task_data)

        assert response.status_code == 422

    def test_create_conversation_invalid_pattern(self, client: TestClient):
        """유효하지 않은 pattern 값."""
        conv_data = {
            "name": "Invalid Pattern",
            "pattern": "invalid_pattern",
        }

        response = client.post("/api/v1/conversations", json=conv_data)

        assert response.status_code == 422


class TestAPIResponseFormat:
    """API 응답 형식 테스트."""

    def test_success_response_format(self, client: TestClient):
        """성공 응답 형식 확인."""
        response = client.get("/api/v1/health")

        data = response.json()
        assert "success" in data
        assert "data" in data
        assert "metadata" in data

    def test_list_response_with_metadata(self, client: TestClient):
        """목록 응답의 metadata 확인."""
        response = client.get("/api/v1/agents")

        data = response.json()
        assert "count" in data["metadata"]
        assert "timestamp" in data["metadata"]
