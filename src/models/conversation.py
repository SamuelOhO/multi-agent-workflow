"""대화(Conversation) 관련 데이터 모델 정의.

이 모듈은 Agent 간 대화 흐름을 관리하는 모델을 정의합니다.
"""

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .message import Message


class ConversationStatus(str, Enum):
    """대화 상태."""

    PENDING = "pending"  # 대기 중
    IN_PROGRESS = "in_progress"  # 진행 중
    COMPLETED = "completed"  # 완료
    FAILED = "failed"  # 실패
    CANCELLED = "cancelled"  # 취소됨
    PAUSED = "paused"  # 일시 중지


class ConversationPattern(str, Enum):
    """대화 패턴 유형."""

    SEQUENTIAL = "sequential"  # 순차 실행
    PARALLEL = "parallel"  # 병렬 실행
    DEBATE = "debate"  # 토론/합의
    HIERARCHICAL = "hierarchical"  # 계층적
    ROUTER = "router"  # 라우터


class ConversationStage(BaseModel):
    """대화 단계 정의.

    대화 흐름의 각 단계를 정의합니다.
    """

    name: str = Field(..., description="단계 이름")
    description: str = Field(default="", description="단계 설명")
    agent_capability: str = Field(
        ..., description="이 단계를 수행할 Agent의 capability"
    )
    agent_id: str | None = Field(
        default=None, description="특정 Agent ID (capability 대신 직접 지정)"
    )
    status: ConversationStatus = Field(
        default=ConversationStatus.PENDING, description="단계 상태"
    )
    input_data: dict[str, Any] = Field(
        default_factory=dict, description="단계 입력 데이터"
    )
    output_data: dict[str, Any] = Field(
        default_factory=dict, description="단계 출력 데이터"
    )
    started_at: datetime | None = Field(default=None, description="시작 시간")
    completed_at: datetime | None = Field(default=None, description="완료 시간")
    error: str | None = Field(default=None, description="에러 메시지 (실패 시)")
    parallel_with: str | None = Field(
        default=None, description="병렬 실행할 다른 단계 이름"
    )

    model_config = {"extra": "forbid"}

    def mark_started(self) -> None:
        """단계 시작 표시."""
        self.status = ConversationStatus.IN_PROGRESS
        self.started_at = datetime.now(UTC)

    def mark_completed(self, output: dict[str, Any] | None = None) -> None:
        """단계 완료 표시."""
        self.status = ConversationStatus.COMPLETED
        self.completed_at = datetime.now(UTC)
        if output:
            self.output_data = output

    def mark_failed(self, error: str) -> None:
        """단계 실패 표시."""
        self.status = ConversationStatus.FAILED
        self.completed_at = datetime.now(UTC)
        self.error = error


class Conversation(BaseModel):
    """대화 세션.

    하나의 작업을 수행하기 위한 Agent 간 대화 전체를 나타냅니다.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="대화 고유 식별자"
    )
    name: str = Field(default="", description="대화 이름")
    description: str = Field(default="", description="대화 설명")
    pattern: ConversationPattern = Field(
        default=ConversationPattern.SEQUENTIAL, description="대화 패턴"
    )
    status: ConversationStatus = Field(
        default=ConversationStatus.PENDING, description="대화 상태"
    )
    stages: list[ConversationStage] = Field(
        default_factory=list, description="대화 단계 목록"
    )
    messages: list[Message] = Field(
        default_factory=list, description="대화 메시지 이력"
    )
    current_stage_index: int = Field(
        default=0, description="현재 진행 중인 단계 인덱스"
    )
    initial_input: dict[str, Any] = Field(
        default_factory=dict, description="초기 입력 데이터 (사용자 요청)"
    )
    final_output: dict[str, Any] = Field(
        default_factory=dict, description="최종 출력 데이터"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="생성 시간"
    )
    started_at: datetime | None = Field(default=None, description="시작 시간")
    completed_at: datetime | None = Field(default=None, description="완료 시간")
    timeout_seconds: int = Field(default=300, description="전체 타임아웃 (초)")
    max_iterations: int = Field(default=3, description="최대 반복 횟수 (토론 패턴용)")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="추가 메타데이터"
    )

    model_config = {"extra": "forbid"}

    def get_current_stage(self) -> ConversationStage | None:
        """현재 단계 반환."""
        if 0 <= self.current_stage_index < len(self.stages):
            return self.stages[self.current_stage_index]
        return None

    def get_stage_by_name(self, name: str) -> ConversationStage | None:
        """이름으로 단계 조회."""
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def add_message(self, message: Message) -> None:
        """메시지 추가."""
        self.messages.append(message)

    def mark_started(self) -> None:
        """대화 시작 표시."""
        self.status = ConversationStatus.IN_PROGRESS
        self.started_at = datetime.now(UTC)

    def mark_completed(self, output: dict[str, Any] | None = None) -> None:
        """대화 완료 표시."""
        self.status = ConversationStatus.COMPLETED
        self.completed_at = datetime.now(UTC)
        if output:
            self.final_output = output

    def mark_failed(self, error: str) -> None:
        """대화 실패 표시."""
        self.status = ConversationStatus.FAILED
        self.completed_at = datetime.now(UTC)
        self.metadata["error"] = error

    def advance_stage(self) -> bool:
        """다음 단계로 진행. 성공 여부 반환."""
        if self.current_stage_index < len(self.stages) - 1:
            self.current_stage_index += 1
            return True
        return False

    def get_completed_stages(self) -> list[ConversationStage]:
        """완료된 단계 목록 반환."""
        return [s for s in self.stages if s.status == ConversationStatus.COMPLETED]

    def get_pending_stages(self) -> list[ConversationStage]:
        """대기 중인 단계 목록 반환."""
        return [s for s in self.stages if s.status == ConversationStatus.PENDING]

    def is_finished(self) -> bool:
        """대화 종료 여부 확인."""
        return self.status in (
            ConversationStatus.COMPLETED,
            ConversationStatus.FAILED,
            ConversationStatus.CANCELLED,
        )


class ConversationResult(BaseModel):
    """대화 실행 결과.

    대화 완료 후 반환되는 결과 정보입니다.
    """

    conversation_id: str = Field(..., description="대화 ID")
    status: ConversationStatus = Field(..., description="최종 상태")
    output: dict[str, Any] = Field(default_factory=dict, description="최종 출력 데이터")
    stages_completed: int = Field(default=0, description="완료된 단계 수")
    stages_total: int = Field(default=0, description="전체 단계 수")
    agents_involved: list[str] = Field(
        default_factory=list, description="참여한 Agent ID 목록"
    )
    messages_count: int = Field(default=0, description="총 메시지 수")
    duration_seconds: float = Field(default=0.0, description="총 소요 시간 (초)")
    error: str | None = Field(default=None, description="에러 메시지 (실패 시)")
    intermediate_results: list[dict[str, Any]] = Field(
        default_factory=list, description="각 단계별 중간 결과"
    )

    model_config = {"extra": "forbid"}

    @classmethod
    def from_conversation(cls, conversation: Conversation) -> "ConversationResult":
        """Conversation으로부터 결과 생성."""
        duration = 0.0
        if conversation.started_at and conversation.completed_at:
            duration = (
                conversation.completed_at - conversation.started_at
            ).total_seconds()

        # 참여한 Agent 목록 추출
        agents = set()
        for message in conversation.messages:
            agents.add(message.sender_id)
            if message.recipient_id:
                agents.add(message.recipient_id)

        # 중간 결과 수집
        intermediate = []
        for stage in conversation.stages:
            if stage.output_data:
                intermediate.append(
                    {
                        "stage": stage.name,
                        "agent_capability": stage.agent_capability,
                        "output": stage.output_data,
                    }
                )

        return cls(
            conversation_id=conversation.id,
            status=conversation.status,
            output=conversation.final_output,
            stages_completed=len(conversation.get_completed_stages()),
            stages_total=len(conversation.stages),
            agents_involved=list(agents),
            messages_count=len(conversation.messages),
            duration_seconds=duration,
            error=conversation.metadata.get("error"),
            intermediate_results=intermediate,
        )
