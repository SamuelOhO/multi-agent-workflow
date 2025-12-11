"""API 스키마 정의.

FastAPI 엔드포인트에서 사용하는 Request/Response 스키마를 정의합니다.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.models import (
    AgentCapability,
    AgentStatus,
    ConversationPattern,
    ConversationStatus,
    MessageType,
)

# =============================================================================
# Common Schemas
# =============================================================================


class APIResponse(BaseModel):
    """표준 API 응답 형식."""

    success: bool = Field(..., description="요청 성공 여부")
    data: Any = Field(default=None, description="응답 데이터")
    error: str | None = Field(default=None, description="에러 메시지")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="추가 메타데이터"
    )


class ErrorResponse(BaseModel):
    """에러 응답 형식."""

    success: bool = Field(default=False, description="항상 False")
    error: str = Field(..., description="에러 메시지")
    error_code: str | None = Field(default=None, description="에러 코드")
    details: dict[str, Any] = Field(default_factory=dict, description="상세 에러 정보")


class PaginatedResponse(BaseModel):
    """페이지네이션된 응답 형식."""

    success: bool = Field(default=True)
    data: list[Any] = Field(default_factory=list, description="데이터 목록")
    total: int = Field(..., description="전체 항목 수")
    page: int = Field(default=1, description="현재 페이지")
    page_size: int = Field(default=20, description="페이지 크기")
    has_next: bool = Field(default=False, description="다음 페이지 존재 여부")


# =============================================================================
# Agent Schemas
# =============================================================================


class AgentCapabilitySchema(BaseModel):
    """Agent Capability 스키마."""

    name: str = Field(..., description="능력 이름")
    description: str = Field(default="", description="능력 설명")
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="입력 스키마"
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict, description="출력 스키마"
    )


class RegisterAgentRequest(BaseModel):
    """Agent 등록 요청."""

    agent_id: str = Field(..., description="Agent 고유 ID")
    name: str = Field(..., description="Agent 이름")
    description: str = Field(default="", description="Agent 설명")
    capabilities: list[AgentCapabilitySchema] = Field(
        ..., description="Agent 능력 목록"
    )
    model: str = Field(
        default="claude-haiku-4-5-20251001", description="사용할 LLM 모델"
    )
    max_tokens: int = Field(default=4096, ge=1, description="최대 토큰 수")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="온도 설정")
    system_prompt: str | None = Field(default=None, description="시스템 프롬프트")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="추가 메타데이터"
    )

    def to_capabilities(self) -> list[AgentCapability]:
        """AgentCapability 객체 목록으로 변환."""
        return [
            AgentCapability(
                name=cap.name,
                description=cap.description,
                input_schema=cap.input_schema,
                output_schema=cap.output_schema,
            )
            for cap in self.capabilities
        ]


class AgentResponse(BaseModel):
    """Agent 정보 응답."""

    agent_id: str = Field(..., description="Agent 고유 ID")
    name: str = Field(..., description="Agent 이름")
    description: str = Field(default="", description="Agent 설명")
    status: AgentStatus = Field(..., description="Agent 상태")
    capabilities: list[str] = Field(
        default_factory=list, description="지원하는 capability 목록"
    )
    load: float = Field(default=0.0, description="현재 부하")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="추가 메타데이터"
    )


class AgentHealthResponse(BaseModel):
    """Agent 상태 체크 응답."""

    agent_id: str = Field(..., description="Agent ID")
    status: str = Field(..., description="상태 (healthy, unhealthy, error)")
    capabilities: list[str] = Field(default_factory=list, description="capability 목록")
    error: str | None = Field(default=None, description="에러 메시지")
    metadata: dict[str, Any] = Field(default_factory=dict, description="추가 정보")


# =============================================================================
# Task Schemas
# =============================================================================


class ConversationStageRequest(BaseModel):
    """대화 단계 요청 스키마."""

    name: str = Field(..., description="단계 이름")
    description: str = Field(default="", description="단계 설명")
    agent_capability: str = Field(..., description="필요한 Agent capability")
    agent_id: str | None = Field(default=None, description="특정 Agent ID (선택)")
    parallel_with: str | None = Field(default=None, description="병렬 실행할 단계 이름")


# =============================================================================
# Pattern-specific Task Schemas
# =============================================================================


class SequentialTaskRequest(BaseModel):
    """Sequential 패턴 작업 요청 - 순차 실행."""

    topic: str = Field(..., description="작업 주제", examples=["Python 웹 스크래퍼 만들기"])
    timeout_seconds: int = Field(default=300, ge=1, le=3600, description="타임아웃 (초)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "topic": "Python으로 뉴스 웹사이트 스크래퍼 만들기",
                "timeout_seconds": 300,
            }
        }
    }


class ParallelTaskRequest(BaseModel):
    """Parallel 패턴 작업 요청 - 병렬 코드 분석."""

    code: str = Field(..., description="분석할 코드", examples=["def calc(x,y): return x/y"])
    timeout_seconds: int = Field(default=300, ge=1, le=3600, description="타임아웃 (초)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "code": "def calculate(x, y):\n    return x / y",
                "timeout_seconds": 300,
            }
        }
    }


class DebateTaskRequest(BaseModel):
    """Debate 패턴 작업 요청 - 토론."""

    topic: str = Field(
        ..., description="토론 주제", examples=["AI가 일자리를 대체하는 것에 대해"]
    )
    max_iterations: int = Field(default=3, ge=1, le=10, description="최대 토론 라운드")
    timeout_seconds: int = Field(default=300, ge=1, le=3600, description="타임아웃 (초)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "topic": "AI가 인간의 일자리를 대체하는 것에 대해 토론해주세요",
                "max_iterations": 3,
                "timeout_seconds": 300,
            }
        }
    }


class RouterTaskRequest(BaseModel):
    """Router 패턴 작업 요청 - 조건 라우팅."""

    question: str = Field(
        ..., description="질문/요청", examples=["이 코드에서 버그를 찾아줘"]
    )
    timeout_seconds: int = Field(default=300, ge=1, le=3600, description="타임아웃 (초)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "이 코드에서 버그를 찾아줘: for i in range(10): print(i",
                "timeout_seconds": 300,
            }
        }
    }


class HierarchicalTaskRequest(BaseModel):
    """Hierarchical 패턴 작업 요청 - 계층 구조."""

    project: str = Field(
        ..., description="프로젝트 설명", examples=["전자상거래 웹사이트 개발"]
    )
    timeout_seconds: int = Field(default=300, ge=1, le=3600, description="타임아웃 (초)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "project": "전자상거래 웹사이트 개발",
                "timeout_seconds": 300,
            }
        }
    }


class CreateTaskRequest(BaseModel):
    """작업 생성 요청."""

    name: str = Field(default="Task", description="작업 이름")
    description: str = Field(default="", description="작업 설명")
    input: dict[str, Any] = Field(..., description="작업 입력 데이터")
    pattern: ConversationPattern = Field(
        default=ConversationPattern.SEQUENTIAL, description="실행 패턴"
    )
    stages: list[ConversationStageRequest] = Field(
        default_factory=list, description="작업 단계 목록"
    )
    timeout_seconds: int = Field(
        default=300, ge=1, le=3600, description="타임아웃 (초)"
    )
    max_iterations: int = Field(
        default=3, ge=1, le=10, description="최대 반복 횟수 (토론 패턴용)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="추가 메타데이터"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Sequential - 순차 실행 (Research → Code → Review)",
                    "value": {
                        "name": "웹 스크래퍼 개발",
                        "pattern": "sequential",
                        "input": {"topic": "Python으로 뉴스 웹사이트 스크래퍼 만들기"},
                        "stages": [
                            {"name": "research", "agent_capability": "web_search"},
                            {"name": "code", "agent_capability": "code_generation"},
                            {"name": "review", "agent_capability": "code_review"},
                        ],
                        "timeout_seconds": 300,
                    },
                },
                {
                    "title": "Parallel - 병렬 실행 (동시 코드 분석)",
                    "value": {
                        "name": "종합 코드 분석",
                        "pattern": "parallel",
                        "input": {"code": "def calculate(x, y): return x/y"},
                        "stages": [
                            {"name": "security", "agent_capability": "security_review"},
                            {
                                "name": "performance",
                                "agent_capability": "performance_review",
                            },
                            {"name": "style", "agent_capability": "style_review"},
                        ],
                        "timeout_seconds": 300,
                    },
                },
                {
                    "title": "Debate - 토론 (낙관론자 vs 비관론자)",
                    "value": {
                        "name": "AI 일자리 대체 토론",
                        "pattern": "debate",
                        "input": {
                            "topic": "AI가 인간의 일자리를 대체하는 것에 대해 토론해주세요"
                        },
                        "stages": [
                            {"name": "optimist_view", "agent_capability": "debate"},
                            {"name": "pessimist_view", "agent_capability": "debate"},
                        ],
                        "timeout_seconds": 300,
                        "max_iterations": 3,
                    },
                },
                {
                    "title": "Router - 조건 라우팅 (질문 유형별 처리)",
                    "value": {
                        "name": "코드 질문 처리",
                        "pattern": "router",
                        "input": {
                            "question": "이 코드에서 버그를 찾아줘: for i in range(10): print(i"
                        },
                        "stages": [
                            {"name": "debug", "agent_capability": "debugging"},
                            {"name": "explain", "agent_capability": "code_explanation"},
                            {"name": "generate", "agent_capability": "code_generation"},
                        ],
                        "timeout_seconds": 300,
                    },
                },
                {
                    "title": "Hierarchical - 계층 구조 (Supervisor → Workers)",
                    "value": {
                        "name": "프로젝트 분석",
                        "pattern": "hierarchical",
                        "input": {"project": "전자상거래 웹사이트 개발"},
                        "stages": [
                            {"name": "supervisor", "agent_capability": "web_search"},
                            {"name": "worker_code", "agent_capability": "code_generation"},
                            {"name": "worker_review", "agent_capability": "code_review"},
                        ],
                        "timeout_seconds": 300,
                    },
                },
            ]
        }
    }


class TaskStatusResponse(BaseModel):
    """작업 상태 응답."""

    task_id: str = Field(..., description="작업 ID (= conversation_id)")
    status: ConversationStatus = Field(..., description="작업 상태")
    pattern: ConversationPattern = Field(..., description="실행 패턴")
    current_stage: str | None = Field(default=None, description="현재 단계")
    progress: dict[str, Any] = Field(default_factory=dict, description="진행 상황")
    created_at: datetime = Field(..., description="생성 시간")
    started_at: datetime | None = Field(default=None, description="시작 시간")
    agents_involved: list[str] = Field(
        default_factory=list, description="참여 Agent 목록"
    )


class TaskResultResponse(BaseModel):
    """작업 결과 응답."""

    task_id: str = Field(..., description="작업 ID")
    status: ConversationStatus = Field(..., description="최종 상태")
    output: dict[str, Any] = Field(default_factory=dict, description="최종 출력")
    stages_completed: int = Field(default=0, description="완료된 단계 수")
    stages_total: int = Field(default=0, description="전체 단계 수")
    agents_involved: list[str] = Field(
        default_factory=list, description="참여 Agent 목록"
    )
    messages_count: int = Field(default=0, description="총 메시지 수")
    duration_seconds: float = Field(default=0.0, description="소요 시간 (초)")
    error: str | None = Field(default=None, description="에러 메시지")
    intermediate_results: list[dict[str, Any]] = Field(
        default_factory=list, description="중간 결과 목록"
    )


# =============================================================================
# Conversation Schemas
# =============================================================================


class CreateConversationRequest(BaseModel):
    """대화 생성 요청."""

    name: str = Field(default="", description="대화 이름")
    description: str = Field(default="", description="대화 설명")
    pattern: ConversationPattern = Field(
        default=ConversationPattern.SEQUENTIAL, description="대화 패턴"
    )
    stages: list[ConversationStageRequest] = Field(
        default_factory=list, description="대화 단계 목록"
    )
    initial_input: dict[str, Any] = Field(
        default_factory=dict, description="초기 입력 데이터"
    )
    timeout_seconds: int = Field(
        default=300, ge=1, le=3600, description="타임아웃 (초)"
    )
    max_iterations: int = Field(default=3, ge=1, le=10, description="최대 반복 횟수")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="추가 메타데이터"
    )


class ConversationResponse(BaseModel):
    """대화 정보 응답."""

    id: str = Field(..., description="대화 ID")
    name: str = Field(default="", description="대화 이름")
    description: str = Field(default="", description="대화 설명")
    pattern: ConversationPattern = Field(..., description="대화 패턴")
    status: ConversationStatus = Field(..., description="대화 상태")
    stages: list[dict[str, Any]] = Field(
        default_factory=list, description="대화 단계 목록"
    )
    current_stage_index: int = Field(default=0, description="현재 단계 인덱스")
    messages_count: int = Field(default=0, description="메시지 수")
    created_at: datetime = Field(..., description="생성 시간")
    started_at: datetime | None = Field(default=None, description="시작 시간")
    completed_at: datetime | None = Field(default=None, description="완료 시간")
    timeout_seconds: int = Field(default=300, description="타임아웃")


class MessageResponse(BaseModel):
    """메시지 응답."""

    id: str = Field(..., description="메시지 ID")
    sender_id: str = Field(..., description="발신자 ID")
    recipient_id: str | None = Field(default=None, description="수신자 ID")
    message_type: MessageType = Field(..., description="메시지 유형")
    content: dict[str, Any] = Field(default_factory=dict, description="메시지 내용")
    timestamp: datetime = Field(..., description="생성 시간")
    correlation_id: str = Field(..., description="상관 ID")


class AddMessageRequest(BaseModel):
    """메시지 추가 요청."""

    sender_id: str = Field(..., description="발신자 ID")
    recipient_id: str | None = Field(default=None, description="수신자 ID")
    message_type: MessageType = Field(
        default=MessageType.TASK, description="메시지 유형"
    )
    content: dict[str, Any] = Field(..., description="메시지 내용")
