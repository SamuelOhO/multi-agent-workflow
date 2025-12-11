"""Agent 관련 데이터 모델 정의.

이 모듈은 Agent의 설정, 상태, 능력(Capability) 등을 정의합니다.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Agent의 현재 상태."""

    ACTIVE = "active"  # 활성 상태, 작업 수행 가능
    INACTIVE = "inactive"  # 비활성 상태
    BUSY = "busy"  # 작업 수행 중
    ERROR = "error"  # 오류 상태
    STARTING = "starting"  # 시작 중
    STOPPING = "stopping"  # 종료 중


class AgentCapability(BaseModel):
    """Agent가 수행할 수 있는 작업(능력) 정의.

    각 Agent는 하나 이상의 Capability를 가지며,
    Orchestrator는 이를 기반으로 적절한 Agent를 선택합니다.
    """

    name: str = Field(..., description="능력 이름 (예: web_search, code_generation)")
    description: str = Field(..., description="능력에 대한 상세 설명")
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="입력 데이터 스키마 (JSON Schema)"
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict, description="출력 데이터 스키마 (JSON Schema)"
    )

    model_config = {"extra": "forbid"}


class AgentConfig(BaseModel):
    """Agent 설정 정보.

    Agent를 생성하고 초기화하는 데 필요한 모든 설정을 포함합니다.
    YAML 설정 파일에서 로드되거나 프로그래밍 방식으로 생성됩니다.
    """

    agent_id: str = Field(..., description="Agent 고유 식별자")
    name: str = Field(..., description="Agent 표시 이름")
    description: str = Field(default="", description="Agent 설명")
    capabilities: list[AgentCapability] = Field(
        default_factory=list, description="Agent가 수행 가능한 작업 목록"
    )
    model: str = Field(
        default="claude-haiku-4-5-20251001", description="사용할 LLM 모델"
    )
    max_tokens: int = Field(default=4096, ge=1, description="최대 응답 토큰 수")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM 온도 설정")
    system_prompt: str | None = Field(
        default=None, description="Agent의 시스템 프롬프트"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="추가 메타데이터"
    )

    model_config = {"extra": "forbid"}

    def get_capability_names(self) -> list[str]:
        """Agent의 모든 capability 이름 목록 반환."""
        return [cap.name for cap in self.capabilities]

    def has_capability(self, capability_name: str) -> bool:
        """특정 capability 보유 여부 확인."""
        return capability_name in self.get_capability_names()


class AgentInfo(BaseModel):
    """Agent 런타임 정보.

    Registry에 등록된 Agent의 현재 상태와 메트릭을 포함합니다.
    """

    agent_id: str = Field(..., description="Agent 고유 식별자")
    name: str = Field(..., description="Agent 표시 이름")
    description: str = Field(default="", description="Agent 설명")
    status: AgentStatus = Field(default=AgentStatus.INACTIVE, description="현재 상태")
    capabilities: list[str] = Field(
        default_factory=list, description="지원하는 capability 이름 목록"
    )
    load: float = Field(
        default=0.0, ge=0.0, le=1.0, description="현재 부하 (0.0 ~ 1.0)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="추가 메타데이터 (모델, 평균 응답 시간 등)"
    )

    model_config = {"extra": "forbid"}

    @classmethod
    def from_config(
        cls, config: AgentConfig, status: AgentStatus = AgentStatus.INACTIVE
    ) -> "AgentInfo":
        """AgentConfig로부터 AgentInfo 생성."""
        return cls(
            agent_id=config.agent_id,
            name=config.name,
            description=config.description,
            status=status,
            capabilities=config.get_capability_names(),
            metadata={
                "model": config.model,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
            },
        )
