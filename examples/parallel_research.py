#!/usr/bin/env python
"""Parallel Research Example - 병렬 처리 예제.

이 예제는 여러 Agent를 병렬로 실행하여 결과를 병합하는 방법을 보여줍니다.
여러 소스에서 동시에 정보를 수집하고 종합하는 시나리오를 구현합니다.

사용법:
    python examples/parallel_research.py
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.base import BaseAgent, SimpleAgent
from src.agents.loader import AgentLoader
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
)
from src.patterns.parallel import MergeStrategy, ParallelPattern


# =============================================================================
# 병렬 실행을 위한 특화 Agent들
# =============================================================================


class NewsSearchAgent(BaseAgent):
    """뉴스 검색을 시뮬레이션하는 Agent."""

    async def process(self, message: Message) -> Message:
        """뉴스 검색 결과를 반환합니다."""
        self.set_busy()

        try:
            query = message.content.get("query") or message.content.get("task", {}).get("query", "")

            # 시뮬레이션된 뉴스 검색 결과
            result = {
                "source": "news",
                "query": query,
                "results": [
                    {"title": f"[뉴스] {query} 관련 최신 동향", "snippet": "최근 발표된 연구에 따르면..."},
                    {"title": f"[뉴스] {query}의 미래 전망", "snippet": "전문가들은 향후 5년 내..."},
                    {"title": f"[뉴스] {query} 시장 분석", "snippet": "글로벌 시장 규모는..."},
                ],
                "count": 3,
            }

            self.activate()
            return self._create_response(message, {"result": result})

        except Exception as e:
            self.set_error(str(e))
            return self._create_error_response(message, str(e))


class AcademicSearchAgent(BaseAgent):
    """학술 자료 검색을 시뮬레이션하는 Agent."""

    async def process(self, message: Message) -> Message:
        """학술 검색 결과를 반환합니다."""
        self.set_busy()

        try:
            query = message.content.get("query") or message.content.get("task", {}).get("query", "")

            # 시뮬레이션된 학술 검색 결과
            result = {
                "source": "academic",
                "query": query,
                "results": [
                    {"title": f"A Survey on {query}", "authors": "Smith et al., 2024", "citations": 150},
                    {"title": f"Advances in {query}: A Review", "authors": "Kim et al., 2023", "citations": 89},
                    {"title": f"Understanding {query}", "authors": "Johnson et al., 2024", "citations": 45},
                ],
                "count": 3,
            }

            self.activate()
            return self._create_response(message, {"result": result})

        except Exception as e:
            self.set_error(str(e))
            return self._create_error_response(message, str(e))


class SocialMediaSearchAgent(BaseAgent):
    """소셜 미디어 검색을 시뮬레이션하는 Agent."""

    async def process(self, message: Message) -> Message:
        """소셜 미디어 검색 결과를 반환합니다."""
        self.set_busy()

        try:
            query = message.content.get("query") or message.content.get("task", {}).get("query", "")

            # 시뮬레이션된 소셜 미디어 검색 결과
            result = {
                "source": "social",
                "query": query,
                "results": [
                    {"platform": "Twitter", "sentiment": "positive", "mentions": 1250},
                    {"platform": "Reddit", "sentiment": "mixed", "mentions": 890},
                    {"platform": "LinkedIn", "sentiment": "positive", "mentions": 450},
                ],
                "trending_topics": [f"#{query}", f"#{query}2024", f"#{query}News"],
                "overall_sentiment": "positive",
            }

            self.activate()
            return self._create_response(message, {"result": result})

        except Exception as e:
            self.set_error(str(e))
            return self._create_error_response(message, str(e))


# =============================================================================
# 설정 생성 함수들
# =============================================================================


def create_news_agent_config() -> AgentConfig:
    """뉴스 검색 Agent 설정."""
    return AgentConfig(
        agent_id="news_search_001",
        name="News Search Agent",
        description="뉴스 소스에서 정보를 검색",
        capabilities=[
            AgentCapability(
                name="news_search",
                description="뉴스 기사 검색",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"results": {"type": "array"}}},
            ),
        ],
    )


def create_academic_agent_config() -> AgentConfig:
    """학술 검색 Agent 설정."""
    return AgentConfig(
        agent_id="academic_search_001",
        name="Academic Search Agent",
        description="학술 자료에서 정보를 검색",
        capabilities=[
            AgentCapability(
                name="academic_search",
                description="학술 논문 검색",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"results": {"type": "array"}}},
            ),
        ],
    )


def create_social_agent_config() -> AgentConfig:
    """소셜 미디어 검색 Agent 설정."""
    return AgentConfig(
        agent_id="social_search_001",
        name="Social Media Search Agent",
        description="소셜 미디어에서 트렌드 검색",
        capabilities=[
            AgentCapability(
                name="social_search",
                description="소셜 미디어 트렌드 검색",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                output_schema={"type": "object", "properties": {"results": {"type": "array"}}},
            ),
        ],
    )


# =============================================================================
# 예제 실행 함수들
# =============================================================================


async def run_parallel_search():
    """병렬 검색 예제를 실행합니다."""
    print("=" * 60)
    print("Parallel Research Example - 병렬 검색")
    print("=" * 60)
    print()

    # 컴포넌트 생성
    registry = AgentRegistry()
    message_bus = InMemoryMessageBus()
    conversation_manager = ConversationManager()

    # 병렬 검색 Agent들 생성 및 등록
    news_agent = NewsSearchAgent(create_news_agent_config())
    academic_agent = AcademicSearchAgent(create_academic_agent_config())
    social_agent = SocialMediaSearchAgent(create_social_agent_config())

    await registry.register(news_agent)
    await registry.register(academic_agent)
    await registry.register(social_agent)

    print("등록된 Agent들:")
    agents = await registry.list_all()
    for agent_info in agents:
        print(f"  - {agent_info.name}")
    print()

    # Orchestrator 생성
    orchestrator = Orchestrator(
        registry=registry,
        message_bus=message_bus,
        conversation_manager=conversation_manager,
    )

    # 검색 작업 정의
    query = "인공지능"
    task = {
        "name": "멀티소스 리서치",
        "description": f"'{query}'에 대한 종합 조사",
        "query": query,
    }

    # 병렬 실행 단계 정의
    stages = [
        ConversationStage(
            name="news_search",
            description="뉴스에서 검색",
            agent_capability="news_search",
        ),
        ConversationStage(
            name="academic_search",
            description="학술 자료에서 검색",
            agent_capability="academic_search",
        ),
        ConversationStage(
            name="social_search",
            description="소셜 미디어에서 검색",
            agent_capability="social_search",
        ),
    ]

    print(f"검색어: '{query}'")
    print(f"검색 소스: {', '.join(s.name for s in stages)}")
    print()
    print("-" * 60)
    print("병렬 검색 실행 중...")
    print("-" * 60)

    # 병렬 패턴으로 실행
    result = await orchestrator.execute(
        task=task,
        pattern=ConversationPattern.PARALLEL,
        stages=stages,
        timeout_seconds=60,
    )

    print()
    print("=" * 60)
    print("검색 결과")
    print("=" * 60)
    print(f"상태: {result.status.value}")
    print(f"소요 시간: {result.duration_seconds:.2f}초")
    print()

    # 결과 출력
    if result.intermediate_results:
        for source, data in result.intermediate_results.items():
            print(f"\n[{source}]")
            if isinstance(data, dict):
                source_name = data.get("source", source)
                results = data.get("results", [])
                print(f"  소스: {source_name}")
                print(f"  결과 수: {len(results)}")
                for i, item in enumerate(results[:3], 1):
                    title = item.get("title") or item.get("platform", "N/A")
                    print(f"  {i}. {title}")

    print()
    print("-" * 60)
    print("병합된 최종 결과:")
    print("-" * 60)
    if result.output:
        total_results = 0
        for _source, data in result.output.items():
            if isinstance(data, dict) and "results" in data:
                total_results += len(data.get("results", []))
        print(f"총 {total_results}개의 결과 수집 완료")


async def run_parallel_with_merge_strategies():
    """다양한 병합 전략을 보여주는 예제."""
    print()
    print("=" * 60)
    print("Parallel Pattern - 병합 전략 예제")
    print("=" * 60)
    print()

    # 컴포넌트 생성
    registry = AgentRegistry()
    message_bus = InMemoryMessageBus()
    conversation_manager = ConversationManager()

    # 간단한 Agent 2개 등록
    config1 = AgentConfig(
        agent_id="agent_a",
        name="Agent A",
        capabilities=[AgentCapability(name="task_a", description="Task A")],
    )
    config2 = AgentConfig(
        agent_id="agent_b",
        name="Agent B",
        capabilities=[AgentCapability(name="task_b", description="Task B")],
    )

    await registry.register(SimpleAgent(config1))
    await registry.register(SimpleAgent(config2))

    # 병합 전략 설명
    print("MergeStrategy 옵션:")
    print("  - DICT: 각 단계의 결과를 딕셔너리로 병합 {stage_name: result}")
    print("  - LIST: 모든 결과를 리스트로 수집 [result1, result2, ...]")
    print("  - FLATTEN: 결과를 평탄화하여 하나의 딕셔너리로 병합")
    print()

    # 각 전략 시연
    for strategy in [MergeStrategy.DICT, MergeStrategy.LIST, MergeStrategy.FLATTEN]:
        print(f"전략: {strategy.value}")

        # ParallelPattern 직접 사용
        pattern = ParallelPattern(
            registry=registry,
            message_bus=message_bus,
            conversation_manager=conversation_manager,
            merge_strategy=strategy,
        )

        print(f"  생성됨: {pattern.name}")
        print()


async def run_debate_example():
    """토론 패턴 예제 (보너스)."""
    print()
    print("=" * 60)
    print("Debate Pattern Example - 토론 패턴")
    print("=" * 60)
    print()

    # 컴포넌트 생성
    registry = AgentRegistry()
    message_bus = InMemoryMessageBus()
    conversation_manager = ConversationManager()

    # 토론용 Agent들 (LLM 필요)
    loader = AgentLoader()
    agents_dir = project_root / "configs" / "agents"

    if agents_dir.exists():
        agents = loader.load_all_from_directory(agents_dir)
        for agent in agents:
            await registry.register(agent)
            print(f"Agent 등록됨: {agent.config.name}")
    else:
        print("Agent 설정 디렉토리를 찾을 수 없습니다.")
        return

    print()

    # Orchestrator 생성
    orchestrator = Orchestrator(
        registry=registry,
        message_bus=message_bus,
        conversation_manager=conversation_manager,
    )

    # 토론 주제
    task = {
        "name": "기술 토론",
        "description": "Python vs JavaScript 토론",
        "query": "웹 개발에서 Python과 JavaScript 중 어느 것이 더 나은가?",
        "topic": "Python vs JavaScript for Web Development",
    }

    # 토론 단계 (최소 2개 Agent 필요)
    stages = [
        ConversationStage(
            name="proposer",
            description="Python 지지 측 의견",
            agent_capability="code_generation",  # CoderAgent
        ),
        ConversationStage(
            name="opponent",
            description="JavaScript 지지 측 의견",
            agent_capability="code_review",  # ReviewerAgent
        ),
    ]

    print(f"토론 주제: {task['topic']}")
    print(f"참가자: {', '.join(s.name for s in stages)}")
    print()

    import os
    if os.getenv("ANTHROPIC_API_KEY"):
        print("토론 시작...")
        try:
            result = await orchestrator.execute(
                task=task,
                pattern=ConversationPattern.DEBATE,
                stages=stages,
                timeout_seconds=180,
            )

            print()
            print("토론 결과:")
            print(f"  상태: {result.status.value}")
            if result.output:
                print(f"  결론: {str(result.output)[:200]}...")
        except Exception as e:
            print(f"토론 실행 중 오류: {e}")
    else:
        print("(ANTHROPIC_API_KEY가 없어 실행을 건너뜁니다)")


async def main():
    """메인 함수 - 예제 선택."""
    print()
    print("Agent Orchestrator - Parallel Research Example")
    print("=" * 60)
    print()
    print("실행할 예제를 선택하세요:")
    print("1. 병렬 검색 예제 (시뮬레이션)")
    print("2. 병합 전략 설명")
    print("3. 토론 패턴 예제 (LLM 필요)")
    print("4. 모든 예제 실행")
    print()

    choice = input("선택 (1/2/3/4, 기본값: 1): ").strip() or "1"

    if choice == "1":
        await run_parallel_search()
    elif choice == "2":
        await run_parallel_with_merge_strategies()
    elif choice == "3":
        await run_debate_example()
    elif choice == "4":
        await run_parallel_search()
        await run_parallel_with_merge_strategies()
        await run_debate_example()
    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    asyncio.run(main())
