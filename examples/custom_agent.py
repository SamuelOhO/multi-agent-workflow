#!/usr/bin/env python
"""Custom Agent Example - 커스텀 Agent 작성 예제.

이 예제는 BaseAgent를 상속하여 커스텀 Agent를 만드는 방법을 보여줍니다.
번역 Agent와 요약 Agent를 예시로 구현합니다.

사용법:
    python examples/custom_agent.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.base import BaseAgent
from src.core.conversation import ConversationManager
from src.core.message_bus import InMemoryMessageBus
from src.core.orchestrator import Orchestrator
from src.core.registry import AgentRegistry
from src.models import (
    AgentCapability,
    AgentConfig,
    ConversationPattern,
    ConversationStage,
    Message,
    MessageType,
)


# =============================================================================
# 커스텀 Agent 구현 예제 1: 번역 Agent
# =============================================================================


class TranslatorAgent(BaseAgent):
    """텍스트 번역을 수행하는 커스텀 Agent.

    이 Agent는 영어-한국어 양방향 번역을 지원합니다.
    """

    def __init__(self, config: AgentConfig):
        """초기화."""
        super().__init__(config)

        # 기본 시스템 프롬프트 설정
        if not config.system_prompt:
            self.config.system_prompt = """당신은 전문 번역가입니다.
주어진 텍스트를 정확하고 자연스럽게 번역하세요.
번역할 때 원문의 의미와 뉘앙스를 최대한 살려주세요.
응답은 JSON 형식으로 제공하세요: {"translated_text": "번역된 텍스트", "source_lang": "원본 언어", "target_lang": "대상 언어"}"""

    async def process(self, message: Message) -> Message:
        """메시지를 처리하고 번역 결과를 반환합니다."""
        self.set_busy()

        try:
            content = message.content
            text = content.get("text") or content.get("query") or content.get("task", {}).get("text", "")
            target_lang = content.get("target_lang", "한국어")
            source_lang = content.get("source_lang", "auto")

            # LLM 호출하여 번역 수행
            prompt = f"""다음 텍스트를 {target_lang}로 번역하세요:

원문: {text}

JSON 형식으로 응답하세요:
{{"translated_text": "번역된 텍스트", "source_lang": "{source_lang}", "target_lang": "{target_lang}"}}"""

            result = await self._call_llm(prompt)

            self.activate()
            return self._create_response(
                message=message,
                content={
                    "result": result,
                    "original_text": text,
                    "target_lang": target_lang,
                },
            )

        except Exception as e:
            self.set_error(str(e))
            return self._create_error_response(message, str(e))


# =============================================================================
# 커스텀 Agent 구현 예제 2: 요약 Agent
# =============================================================================


class SummarizerAgent(BaseAgent):
    """텍스트 요약을 수행하는 커스텀 Agent.

    긴 텍스트를 지정된 길이로 요약합니다.
    """

    def __init__(self, config: AgentConfig):
        """초기화."""
        super().__init__(config)

        if not config.system_prompt:
            self.config.system_prompt = """당신은 전문 텍스트 요약가입니다.
주어진 텍스트의 핵심 내용을 파악하고 간결하게 요약하세요.
중요한 정보는 빠뜨리지 않으면서도 불필요한 부분은 제거하세요."""

    async def process(self, message: Message) -> Message:
        """메시지를 처리하고 요약 결과를 반환합니다."""
        self.set_busy()

        try:
            content = message.content
            text = content.get("text") or content.get("query") or content.get("task", {}).get("text", "")
            max_length = content.get("max_length", 200)
            style = content.get("style", "bullet_points")  # bullet_points, paragraph, keywords

            # LLM 호출하여 요약 수행
            style_instruction = {
                "bullet_points": "핵심 포인트를 글머리 기호로 나열하세요.",
                "paragraph": "하나의 문단으로 요약하세요.",
                "keywords": "핵심 키워드만 추출하세요.",
            }.get(style, "간결하게 요약하세요.")

            prompt = f"""다음 텍스트를 {max_length}자 이내로 요약하세요.
{style_instruction}

원문:
{text}

요약:"""

            result = await self._call_llm(prompt)

            self.activate()
            return self._create_response(
                message=message,
                content={
                    "result": result,
                    "summary": result,
                    "original_length": len(text),
                    "summary_length": len(result),
                    "style": style,
                },
            )

        except Exception as e:
            self.set_error(str(e))
            return self._create_error_response(message, str(e))


# =============================================================================
# 커스텀 Agent 구현 예제 3: 감정 분석 Agent (LLM 없이 간단한 로직)
# =============================================================================


class SentimentAgent(BaseAgent):
    """간단한 감정 분석을 수행하는 Agent.

    이 예제는 LLM 없이 간단한 규칙 기반으로 감정을 분석합니다.
    실제 프로덕션에서는 LLM이나 ML 모델을 사용하세요.
    """

    # 간단한 감정 키워드 사전
    POSITIVE_WORDS = {"좋다", "훌륭하다", "최고", "감사", "행복", "좋아", "멋지다", "great", "good", "excellent", "happy", "love"}
    NEGATIVE_WORDS = {"나쁘다", "싫다", "최악", "슬프다", "화나다", "실망", "bad", "terrible", "hate", "sad", "angry"}

    async def process(self, message: Message) -> Message:
        """메시지를 처리하고 감정 분석 결과를 반환합니다."""
        self.set_busy()

        try:
            content = message.content
            text = content.get("text") or content.get("query") or content.get("task", {}).get("text", "")
            text_lower = text.lower()

            # 간단한 키워드 기반 분석
            positive_count = sum(1 for word in self.POSITIVE_WORDS if word in text_lower)
            negative_count = sum(1 for word in self.NEGATIVE_WORDS if word in text_lower)

            if positive_count > negative_count:
                sentiment = "positive"
                confidence = min(positive_count / (positive_count + negative_count + 1), 0.9)
            elif negative_count > positive_count:
                sentiment = "negative"
                confidence = min(negative_count / (positive_count + negative_count + 1), 0.9)
            else:
                sentiment = "neutral"
                confidence = 0.5

            self.activate()
            return self._create_response(
                message=message,
                content={
                    "result": {
                        "sentiment": sentiment,
                        "confidence": round(confidence, 2),
                        "positive_indicators": positive_count,
                        "negative_indicators": negative_count,
                    },
                    "text_analyzed": text[:100] + "..." if len(text) > 100 else text,
                },
            )

        except Exception as e:
            self.set_error(str(e))
            return self._create_error_response(message, str(e))


# =============================================================================
# Agent 설정 및 등록
# =============================================================================


def create_translator_config() -> AgentConfig:
    """번역 Agent 설정을 생성합니다."""
    return AgentConfig(
        agent_id="translator_001",
        name="Translator Agent",
        description="텍스트 번역을 수행하는 Agent",
        capabilities=[
            AgentCapability(
                name="translation",
                description="텍스트를 다른 언어로 번역",
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "번역할 텍스트"},
                        "target_lang": {"type": "string", "description": "대상 언어"},
                    },
                    "required": ["text"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "translated_text": {"type": "string"},
                        "source_lang": {"type": "string"},
                        "target_lang": {"type": "string"},
                    },
                },
            ),
        ],
        model="claude-haiku-4-5-20251001",
        temperature=0.3,  # 번역은 낮은 temperature가 좋음
    )


def create_summarizer_config() -> AgentConfig:
    """요약 Agent 설정을 생성합니다."""
    return AgentConfig(
        agent_id="summarizer_001",
        name="Summarizer Agent",
        description="텍스트 요약을 수행하는 Agent",
        capabilities=[
            AgentCapability(
                name="summarization",
                description="긴 텍스트를 요약",
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "max_length": {"type": "integer", "default": 200},
                        "style": {"type": "string", "enum": ["bullet_points", "paragraph", "keywords"]},
                    },
                    "required": ["text"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "original_length": {"type": "integer"},
                        "summary_length": {"type": "integer"},
                    },
                },
            ),
        ],
        model="claude-haiku-4-5-20251001",
        temperature=0.5,
    )


def create_sentiment_config() -> AgentConfig:
    """감정 분석 Agent 설정을 생성합니다."""
    return AgentConfig(
        agent_id="sentiment_001",
        name="Sentiment Agent",
        description="텍스트의 감정을 분석하는 Agent",
        capabilities=[
            AgentCapability(
                name="sentiment_analysis",
                description="텍스트의 감정(긍정/부정/중립) 분석",
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "분석할 텍스트"},
                    },
                    "required": ["text"],
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                        "confidence": {"type": "number"},
                    },
                },
            ),
        ],
    )


# =============================================================================
# 예제 실행
# =============================================================================


async def run_custom_agents_example():
    """커스텀 Agent들을 사용하는 예제를 실행합니다."""
    print("=" * 60)
    print("Custom Agent Example")
    print("=" * 60)
    print()

    # 컴포넌트 생성
    registry = AgentRegistry()
    message_bus = InMemoryMessageBus()
    conversation_manager = ConversationManager()

    # 커스텀 Agent 생성 및 등록
    translator = TranslatorAgent(create_translator_config())
    summarizer = SummarizerAgent(create_summarizer_config())
    sentiment = SentimentAgent(create_sentiment_config())

    await registry.register(translator)
    await registry.register(summarizer)
    await registry.register(sentiment)

    print("등록된 커스텀 Agent들:")
    agents = await registry.list_all()
    for agent_info in agents:
        print(f"  - {agent_info.name} ({agent_info.agent_id})")
        print(f"    Capabilities: {', '.join(agent_info.capabilities)}")
    print()

    # Orchestrator 생성
    orchestrator = Orchestrator(
        registry=registry,
        message_bus=message_bus,
        conversation_manager=conversation_manager,
    )

    # 예제 1: 감정 분석 (LLM 없이 실행 가능)
    print("-" * 60)
    print("예제 1: 감정 분석")
    print("-" * 60)

    task1 = {
        "name": "감정 분석",
        "text": "이 제품은 정말 훌륭합니다! 최고의 선택이었어요. 감사합니다!",
    }

    stages1 = [
        ConversationStage(
            name="sentiment",
            description="감정 분석",
            agent_capability="sentiment_analysis",
        ),
    ]

    result1 = await orchestrator.execute(
        task=task1,
        pattern=ConversationPattern.SEQUENTIAL,
        stages=stages1,
        timeout_seconds=30,
    )

    print(f"텍스트: {task1['text']}")
    print(f"결과: {result1.output}")
    print()

    # 예제 2: 번역 + 요약 파이프라인 (LLM 필요)
    print("-" * 60)
    print("예제 2: 번역 -> 요약 파이프라인")
    print("-" * 60)
    print("(참고: 이 예제는 ANTHROPIC_API_KEY가 필요합니다)")

    task2 = {
        "name": "번역 후 요약",
        "text": """Artificial Intelligence is transforming the way we live and work.
        From healthcare to transportation, AI applications are becoming increasingly prevalent.
        Machine learning algorithms can now diagnose diseases, drive cars, and even create art.
        However, with these advances come important ethical considerations that society must address.""",
        "target_lang": "한국어",
    }

    # 이 파이프라인은 LLM이 필요하므로 API 키가 없으면 스킵
    import os
    if os.getenv("ANTHROPIC_API_KEY"):
        stages2 = [
            ConversationStage(
                name="translate",
                description="영어 텍스트를 한국어로 번역",
                agent_capability="translation",
            ),
            ConversationStage(
                name="summarize",
                description="번역된 텍스트 요약",
                agent_capability="summarization",
            ),
        ]

        try:
            result2 = await orchestrator.execute(
                task=task2,
                pattern=ConversationPattern.SEQUENTIAL,
                stages=stages2,
                timeout_seconds=120,
            )
            print(f"원문: {task2['text'][:50]}...")
            print(f"결과: {result2.output}")
        except Exception as e:
            print(f"실행 중 오류 발생: {e}")
    else:
        print("ANTHROPIC_API_KEY가 설정되지 않아 스킵합니다.")

    print()
    print("=" * 60)
    print("커스텀 Agent 예제 완료!")
    print("=" * 60)


async def main():
    """메인 함수."""
    await run_custom_agents_example()


if __name__ == "__main__":
    asyncio.run(main())
