"""데이터 모델 단위 테스트."""

import pytest

from src.models import (
    # Agent models
    AgentCapability,
    AgentConfig,
    AgentInfo,
    AgentStatus,
    # Conversation models
    Conversation,
    ConversationPattern,
    ConversationResult,
    ConversationStage,
    ConversationStatus,
    # Message models
    Message,
    MessageContext,
    MessageType,
)


class TestAgentModels:
    """Agent 모델 테스트."""

    def test_agent_capability_creation(self):
        """AgentCapability 생성 테스트."""
        capability = AgentCapability(
            name="web_search",
            description="웹에서 정보를 검색합니다",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            output_schema={
                "type": "object",
                "properties": {"results": {"type": "array"}},
            },
        )

        assert capability.name == "web_search"
        assert capability.description == "웹에서 정보를 검색합니다"
        assert "query" in capability.input_schema["properties"]

    def test_agent_capability_defaults(self):
        """AgentCapability 기본값 테스트."""
        capability = AgentCapability(
            name="test",
            description="테스트",
        )

        assert capability.input_schema == {}
        assert capability.output_schema == {}

    def test_agent_config_creation(self):
        """AgentConfig 생성 테스트."""
        capability = AgentCapability(name="code_gen", description="코드 생성")
        config = AgentConfig(
            agent_id="coder_001",
            name="Coder Agent",
            description="코드 생성 전문 Agent",
            capabilities=[capability],
            model="claude-haiku-4-5-20251001",
            max_tokens=8192,
            temperature=0.5,
        )

        assert config.agent_id == "coder_001"
        assert config.name == "Coder Agent"
        assert len(config.capabilities) == 1
        assert config.max_tokens == 8192

    def test_agent_config_defaults(self):
        """AgentConfig 기본값 테스트."""
        config = AgentConfig(
            agent_id="test_001",
            name="Test Agent",
        )

        assert config.model == "claude-haiku-4-5-20251001"
        assert config.max_tokens == 4096
        assert config.temperature == 0.7
        assert config.capabilities == []

    def test_agent_config_get_capability_names(self):
        """AgentConfig.get_capability_names() 테스트."""
        config = AgentConfig(
            agent_id="test",
            name="Test",
            capabilities=[
                AgentCapability(name="cap1", description="Capability 1"),
                AgentCapability(name="cap2", description="Capability 2"),
            ],
        )

        names = config.get_capability_names()
        assert names == ["cap1", "cap2"]

    def test_agent_config_has_capability(self):
        """AgentConfig.has_capability() 테스트."""
        config = AgentConfig(
            agent_id="test",
            name="Test",
            capabilities=[
                AgentCapability(name="web_search", description="검색"),
            ],
        )

        assert config.has_capability("web_search") is True
        assert config.has_capability("code_gen") is False

    def test_agent_status_enum(self):
        """AgentStatus Enum 테스트."""
        assert AgentStatus.ACTIVE == "active"
        assert AgentStatus.BUSY == "busy"
        assert AgentStatus.ERROR == "error"

    def test_agent_info_creation(self):
        """AgentInfo 생성 테스트."""
        info = AgentInfo(
            agent_id="agent_001",
            name="Test Agent",
            status=AgentStatus.ACTIVE,
            capabilities=["search", "summarize"],
            load=0.5,
        )

        assert info.agent_id == "agent_001"
        assert info.status == AgentStatus.ACTIVE
        assert info.load == 0.5
        assert "search" in info.capabilities

    def test_agent_info_from_config(self):
        """AgentInfo.from_config() 테스트."""
        config = AgentConfig(
            agent_id="test_001",
            name="Test Agent",
            description="테스트용 Agent",
            capabilities=[
                AgentCapability(name="cap1", description="Capability 1"),
            ],
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
        )

        info = AgentInfo.from_config(config, status=AgentStatus.ACTIVE)

        assert info.agent_id == "test_001"
        assert info.name == "Test Agent"
        assert info.status == AgentStatus.ACTIVE
        assert "cap1" in info.capabilities
        assert info.metadata["model"] == "claude-haiku-4-5-20251001"


class TestMessageModels:
    """Message 모델 테스트."""

    def test_message_type_enum(self):
        """MessageType Enum 테스트."""
        assert MessageType.TASK == "task"
        assert MessageType.RESPONSE == "response"
        assert MessageType.ERROR == "error"

    def test_message_context_creation(self):
        """MessageContext 생성 테스트."""
        context = MessageContext(
            conversation_id="conv_001",
            parent_message_id="msg_001",
            stage="research",
            history=[{"role": "user", "content": "검색해줘"}],
        )

        assert context.conversation_id == "conv_001"
        assert context.parent_message_id == "msg_001"
        assert len(context.history) == 1

    def test_message_context_defaults(self):
        """MessageContext 기본값 테스트."""
        context = MessageContext()

        assert context.conversation_id is None
        assert context.parent_message_id is None
        assert context.history == []
        assert context.metadata == {}

    def test_message_creation(self):
        """Message 생성 테스트."""
        message = Message(
            sender_id="orchestrator",
            recipient_id="researcher_001",
            message_type=MessageType.TASK,
            content={"task": "AI 트렌드 조사"},
        )

        assert message.sender_id == "orchestrator"
        assert message.recipient_id == "researcher_001"
        assert message.message_type == MessageType.TASK
        assert message.id is not None
        assert message.correlation_id is not None

    def test_message_auto_generated_fields(self):
        """Message 자동 생성 필드 테스트."""
        message = Message(
            sender_id="test",
            recipient_id="test2",
            message_type=MessageType.NOTIFICATION,
            content={},
        )

        # ID와 timestamp가 자동 생성되는지 확인
        assert message.id is not None
        assert len(message.id) == 36  # UUID 형식
        assert message.timestamp is not None
        assert message.correlation_id is not None

    def test_message_create_task(self):
        """Message.create_task() 헬퍼 테스트."""
        message = Message.create_task(
            sender_id="orchestrator",
            recipient_id="coder_001",
            content={"code": "print('hello')"},
        )

        assert message.message_type == MessageType.TASK
        assert message.sender_id == "orchestrator"
        assert message.recipient_id == "coder_001"

    def test_message_create_response(self):
        """Message.create_response() 헬퍼 테스트."""
        message = Message.create_response(
            sender_id="coder_001",
            recipient_id="orchestrator",
            content={"result": "success"},
            correlation_id="corr_123",
            parent_message_id="msg_001",
        )

        assert message.message_type == MessageType.RESPONSE
        assert message.correlation_id == "corr_123"
        assert message.context.parent_message_id == "msg_001"

    def test_message_create_error(self):
        """Message.create_error() 헬퍼 테스트."""
        message = Message.create_error(
            sender_id="agent_001",
            recipient_id="orchestrator",
            error_message="처리 중 오류 발생",
            error_code="E001",
        )

        assert message.message_type == MessageType.ERROR
        assert message.content["error"] == "처리 중 오류 발생"
        assert message.content["error_code"] == "E001"

    def test_message_is_broadcast(self):
        """Message.is_broadcast() 테스트."""
        broadcast = Message(
            sender_id="orchestrator",
            recipient_id=None,
            message_type=MessageType.NOTIFICATION,
            content={"notice": "시스템 점검"},
        )
        direct = Message(
            sender_id="orchestrator",
            recipient_id="agent_001",
            message_type=MessageType.TASK,
            content={},
        )

        assert broadcast.is_broadcast() is True
        assert direct.is_broadcast() is False

    def test_message_is_response_to(self):
        """Message.is_response_to() 테스트."""
        original = Message(
            sender_id="orchestrator",
            recipient_id="agent_001",
            message_type=MessageType.TASK,
            content={},
            correlation_id="corr_123",
        )
        response = Message(
            sender_id="agent_001",
            recipient_id="orchestrator",
            message_type=MessageType.RESPONSE,
            content={},
            correlation_id="corr_123",
        )
        unrelated = Message(
            sender_id="agent_002",
            recipient_id="orchestrator",
            message_type=MessageType.RESPONSE,
            content={},
            correlation_id="corr_456",
        )

        assert response.is_response_to(original) is True
        assert unrelated.is_response_to(original) is False


class TestConversationModels:
    """Conversation 모델 테스트."""

    def test_conversation_status_enum(self):
        """ConversationStatus Enum 테스트."""
        assert ConversationStatus.PENDING == "pending"
        assert ConversationStatus.IN_PROGRESS == "in_progress"
        assert ConversationStatus.COMPLETED == "completed"

    def test_conversation_pattern_enum(self):
        """ConversationPattern Enum 테스트."""
        assert ConversationPattern.SEQUENTIAL == "sequential"
        assert ConversationPattern.PARALLEL == "parallel"
        assert ConversationPattern.DEBATE == "debate"

    def test_conversation_stage_creation(self):
        """ConversationStage 생성 테스트."""
        stage = ConversationStage(
            name="research",
            description="기술 조사",
            agent_capability="web_search",
        )

        assert stage.name == "research"
        assert stage.agent_capability == "web_search"
        assert stage.status == ConversationStatus.PENDING

    def test_conversation_stage_mark_methods(self):
        """ConversationStage 상태 변경 메서드 테스트."""
        stage = ConversationStage(
            name="test",
            description="테스트",
            agent_capability="test_cap",
        )

        # 시작
        stage.mark_started()
        assert stage.status == ConversationStatus.IN_PROGRESS
        assert stage.started_at is not None

        # 완료
        stage.mark_completed({"result": "success"})
        assert stage.status == ConversationStatus.COMPLETED
        assert stage.completed_at is not None
        assert stage.output_data["result"] == "success"

    def test_conversation_stage_mark_failed(self):
        """ConversationStage.mark_failed() 테스트."""
        stage = ConversationStage(
            name="test",
            description="테스트",
            agent_capability="test_cap",
        )

        stage.mark_failed("처리 실패")
        assert stage.status == ConversationStatus.FAILED
        assert stage.error == "처리 실패"

    def test_conversation_creation(self):
        """Conversation 생성 테스트."""
        conv = Conversation(
            name="소프트웨어 개발",
            description="웹 스크래퍼 개발",
            pattern=ConversationPattern.SEQUENTIAL,
            stages=[
                ConversationStage(
                    name="research", description="조사", agent_capability="web_search"
                ),
                ConversationStage(
                    name="code", description="구현", agent_capability="code_gen"
                ),
            ],
            initial_input={"task": "웹 스크래퍼 만들기"},
        )

        assert conv.name == "소프트웨어 개발"
        assert conv.pattern == ConversationPattern.SEQUENTIAL
        assert len(conv.stages) == 2
        assert conv.status == ConversationStatus.PENDING

    def test_conversation_get_current_stage(self):
        """Conversation.get_current_stage() 테스트."""
        conv = Conversation(
            name="test",
            stages=[
                ConversationStage(
                    name="stage1", description="1", agent_capability="cap1"
                ),
                ConversationStage(
                    name="stage2", description="2", agent_capability="cap2"
                ),
            ],
        )

        current = conv.get_current_stage()
        assert current is not None
        assert current.name == "stage1"

    def test_conversation_get_stage_by_name(self):
        """Conversation.get_stage_by_name() 테스트."""
        conv = Conversation(
            name="test",
            stages=[
                ConversationStage(
                    name="research", description="조사", agent_capability="search"
                ),
                ConversationStage(
                    name="code", description="코딩", agent_capability="code_gen"
                ),
            ],
        )

        stage = conv.get_stage_by_name("code")
        assert stage is not None
        assert stage.agent_capability == "code_gen"

        assert conv.get_stage_by_name("nonexistent") is None

    def test_conversation_advance_stage(self):
        """Conversation.advance_stage() 테스트."""
        conv = Conversation(
            name="test",
            stages=[
                ConversationStage(
                    name="stage1", description="1", agent_capability="cap1"
                ),
                ConversationStage(
                    name="stage2", description="2", agent_capability="cap2"
                ),
            ],
        )

        assert conv.current_stage_index == 0
        assert conv.advance_stage() is True
        assert conv.current_stage_index == 1
        assert conv.advance_stage() is False  # 마지막 단계

    def test_conversation_mark_methods(self):
        """Conversation 상태 변경 메서드 테스트."""
        conv = Conversation(name="test")

        conv.mark_started()
        assert conv.status == ConversationStatus.IN_PROGRESS
        assert conv.started_at is not None

        conv.mark_completed({"result": "done"})
        assert conv.status == ConversationStatus.COMPLETED
        assert conv.final_output["result"] == "done"

    def test_conversation_is_finished(self):
        """Conversation.is_finished() 테스트."""
        conv = Conversation(name="test")

        assert conv.is_finished() is False

        conv.status = ConversationStatus.COMPLETED
        assert conv.is_finished() is True

        conv.status = ConversationStatus.FAILED
        assert conv.is_finished() is True

        conv.status = ConversationStatus.CANCELLED
        assert conv.is_finished() is True

    def test_conversation_add_message(self):
        """Conversation.add_message() 테스트."""
        conv = Conversation(name="test")
        message = Message(
            sender_id="agent",
            recipient_id="orchestrator",
            message_type=MessageType.RESPONSE,
            content={},
        )

        conv.add_message(message)
        assert len(conv.messages) == 1
        assert conv.messages[0] == message

    def test_conversation_result_creation(self):
        """ConversationResult 생성 테스트."""
        result = ConversationResult(
            conversation_id="conv_001",
            status=ConversationStatus.COMPLETED,
            output={"summary": "완료"},
            stages_completed=3,
            stages_total=3,
            agents_involved=["agent1", "agent2"],
            duration_seconds=45.5,
        )

        assert result.conversation_id == "conv_001"
        assert result.status == ConversationStatus.COMPLETED
        assert result.stages_completed == 3

    def test_conversation_result_from_conversation(self):
        """ConversationResult.from_conversation() 테스트."""
        conv = Conversation(
            name="test",
            stages=[
                ConversationStage(name="s1", description="1", agent_capability="c1"),
                ConversationStage(name="s2", description="2", agent_capability="c2"),
            ],
        )

        # 시작 및 완료 시뮬레이션
        conv.mark_started()
        conv.stages[0].mark_completed({"data": "result1"})
        conv.mark_completed({"final": "done"})

        # 메시지 추가
        conv.add_message(
            Message(
                sender_id="orchestrator",
                recipient_id="agent1",
                message_type=MessageType.TASK,
                content={},
            )
        )

        result = ConversationResult.from_conversation(conv)

        assert result.conversation_id == conv.id
        assert result.status == ConversationStatus.COMPLETED
        assert result.stages_completed == 1
        assert result.stages_total == 2
        assert result.messages_count == 1
        assert len(result.intermediate_results) == 1


class TestModelValidation:
    """모델 유효성 검사 테스트."""

    def test_agent_config_temperature_validation(self):
        """AgentConfig temperature 범위 검증 테스트."""
        # 유효한 범위
        config = AgentConfig(
            agent_id="test",
            name="Test",
            temperature=1.5,
        )
        assert config.temperature == 1.5

        # 범위 초과
        with pytest.raises(ValueError):
            AgentConfig(
                agent_id="test",
                name="Test",
                temperature=2.5,
            )

        # 음수
        with pytest.raises(ValueError):
            AgentConfig(
                agent_id="test",
                name="Test",
                temperature=-0.1,
            )

    def test_agent_config_max_tokens_validation(self):
        """AgentConfig max_tokens 검증 테스트."""
        # 유효한 값
        config = AgentConfig(
            agent_id="test",
            name="Test",
            max_tokens=100,
        )
        assert config.max_tokens == 100

        # 0 이하
        with pytest.raises(ValueError):
            AgentConfig(
                agent_id="test",
                name="Test",
                max_tokens=0,
            )

    def test_agent_info_load_validation(self):
        """AgentInfo load 범위 검증 테스트."""
        # 유효한 범위
        info = AgentInfo(
            agent_id="test",
            name="Test",
            load=0.5,
        )
        assert info.load == 0.5

        # 범위 초과
        with pytest.raises(ValueError):
            AgentInfo(
                agent_id="test",
                name="Test",
                load=1.5,
            )
