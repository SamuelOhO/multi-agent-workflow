"""메시지 관련 데이터 모델 정의.

이 모듈은 Agent 간 통신에 사용되는 메시지 모델을 정의합니다.
"""

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """메시지 유형."""

    TASK = "task"  # 작업 요청
    RESPONSE = "response"  # 작업 응답
    QUERY = "query"  # 정보 조회 요청
    NOTIFICATION = "notification"  # 알림
    ERROR = "error"  # 에러 메시지
    SYSTEM = "system"  # 시스템 메시지


class MessageContext(BaseModel):
    """메시지 컨텍스트 정보.

    대화 맥락과 관련된 추가 정보를 포함합니다.
    """

    conversation_id: str | None = Field(default=None, description="대화 ID")
    parent_message_id: str | None = Field(
        default=None, description="부모 메시지 ID (응답인 경우)"
    )
    stage: str | None = Field(default=None, description="현재 대화 단계")
    history: list[dict[str, Any]] = Field(
        default_factory=list, description="이전 메시지 요약/이력"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="추가 컨텍스트 데이터"
    )

    model_config = {"extra": "forbid"}


class Message(BaseModel):
    """Agent 간 통신 메시지.

    모든 Agent 간 통신은 이 메시지 형식을 사용합니다.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="메시지 고유 식별자"
    )
    sender_id: str = Field(..., description="발신자 Agent ID")
    recipient_id: str | None = Field(
        default=None, description="수신자 Agent ID (None이면 브로드캐스트)"
    )
    message_type: MessageType = Field(..., description="메시지 유형")
    content: dict[str, Any] = Field(default_factory=dict, description="메시지 내용")
    context: MessageContext = Field(
        default_factory=MessageContext, description="메시지 컨텍스트"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="메시지 생성 시간"
    )
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="관련 대화 추적용 ID"
    )

    model_config = {"extra": "forbid"}

    @classmethod
    def create_task(
        cls,
        sender_id: str,
        recipient_id: str | None,
        content: dict[str, Any],
        correlation_id: str | None = None,
        context: MessageContext | None = None,
    ) -> "Message":
        """작업 요청 메시지 생성 헬퍼."""
        return cls(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.TASK,
            content=content,
            correlation_id=correlation_id or str(uuid.uuid4()),
            context=context or MessageContext(),
        )

    @classmethod
    def create_response(
        cls,
        sender_id: str,
        recipient_id: str,
        content: dict[str, Any],
        correlation_id: str,
        parent_message_id: str | None = None,
        context: MessageContext | None = None,
    ) -> "Message":
        """응답 메시지 생성 헬퍼."""
        if context is None:
            context = MessageContext(parent_message_id=parent_message_id)
        elif parent_message_id:
            context.parent_message_id = parent_message_id

        return cls(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.RESPONSE,
            content=content,
            correlation_id=correlation_id,
            context=context,
        )

    @classmethod
    def create_error(
        cls,
        sender_id: str,
        recipient_id: str | None,
        error_message: str,
        error_code: str | None = None,
        correlation_id: str | None = None,
    ) -> "Message":
        """에러 메시지 생성 헬퍼."""
        content = {"error": error_message}
        if error_code:
            content["error_code"] = error_code

        return cls(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.ERROR,
            content=content,
            correlation_id=correlation_id or str(uuid.uuid4()),
        )

    def is_broadcast(self) -> bool:
        """브로드캐스트 메시지 여부 확인."""
        return self.recipient_id is None

    def is_response_to(self, message: "Message") -> bool:
        """특정 메시지에 대한 응답인지 확인."""
        return (
            self.message_type == MessageType.RESPONSE
            and self.correlation_id == message.correlation_id
        )
