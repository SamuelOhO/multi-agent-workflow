#!/usr/bin/env python
"""Simple Pipeline Example - 기본 사용 예제.

이 예제는 Agent Orchestrator의 기본적인 사용 방법을 보여줍니다.
순차적(Sequential) 패턴을 사용하여 Research -> Code -> Review 파이프라인을 실행합니다.

사용법:
    python examples/simple_pipeline.py
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.loader import AgentLoader
from src.core.conversation import ConversationManager
from src.core.message_bus import InMemoryMessageBus
from src.core.orchestrator import Orchestrator
from src.core.registry import AgentRegistry
from src.models import ConversationPattern, ConversationStage


async def setup_orchestrator() -> Orchestrator:
    """Orchestrator와 필요한 컴포넌트들을 설정합니다."""
    # 핵심 컴포넌트 생성
    registry = AgentRegistry()
    message_bus = InMemoryMessageBus()
    conversation_manager = ConversationManager()

    # Agent 로더로 설정 파일에서 Agent들 로드
    loader = AgentLoader()
    agents_dir = project_root / "configs" / "agents"

    if agents_dir.exists():
        agents = loader.load_all_from_directory(agents_dir)
        for agent in agents:
            await registry.register(agent)
            print(f"Agent 등록됨: {agent.config.name} ({agent.config.agent_id})")
    else:
        print(f"경고: Agent 설정 디렉토리를 찾을 수 없습니다: {agents_dir}")

    # Orchestrator 생성
    orchestrator = Orchestrator(
        registry=registry,
        message_bus=message_bus,
        conversation_manager=conversation_manager,
    )

    return orchestrator


async def run_simple_pipeline():
    """간단한 순차 파이프라인을 실행합니다."""
    print("=" * 60)
    print("Simple Pipeline Example")
    print("=" * 60)
    print()

    # Orchestrator 설정
    orchestrator = await setup_orchestrator()

    # 작업 정의
    task = {
        "name": "웹 스크래퍼 개발",
        "description": "간단한 웹 스크래퍼를 개발합니다.",
        "query": "Python으로 뉴스 사이트를 스크래핑하는 코드를 작성해줘",
        "requirements": [
            "requests 라이브러리 사용",
            "BeautifulSoup으로 파싱",
            "제목과 본문 추출",
        ],
    }

    # 파이프라인 단계 정의
    stages = [
        ConversationStage(
            name="research",
            description="기술 조사 및 요구사항 분석",
            agent_capability="web_search",  # ResearchAgent
        ),
        ConversationStage(
            name="implementation",
            description="코드 구현",
            agent_capability="code_generation",  # CoderAgent
        ),
        ConversationStage(
            name="review",
            description="코드 리뷰 및 개선 제안",
            agent_capability="code_review",  # ReviewerAgent
        ),
    ]

    print(f"작업: {task['name']}")
    print(f"설명: {task['description']}")
    print(f"단계: {' -> '.join(s.name for s in stages)}")
    print()
    print("-" * 60)
    print("실행 중...")
    print("-" * 60)

    try:
        # 파이프라인 실행
        result = await orchestrator.execute(
            task=task,
            pattern=ConversationPattern.SEQUENTIAL,
            stages=stages,
            timeout_seconds=300,
        )

        print()
        print("=" * 60)
        print("실행 결과")
        print("=" * 60)
        print(f"상태: {result.status.value}")
        print(f"소요 시간: {result.duration_seconds:.2f}초")
        print()

        if result.intermediate_results:
            print("단계별 결과:")
            for stage_name, stage_result in result.intermediate_results.items():
                print(f"\n[{stage_name}]")
                if isinstance(stage_result, dict):
                    for key, value in stage_result.items():
                        print(f"  {key}: {str(value)[:100]}...")
                else:
                    print(f"  {str(stage_result)[:200]}...")

        print()
        print("최종 결과:")
        if result.output:
            for key, value in result.output.items():
                print(f"  {key}: {str(value)[:200]}...")

    except Exception as e:
        print(f"오류 발생: {e}")
        raise


async def run_quick_task():
    """단일 Agent로 빠른 작업을 실행합니다."""
    print("=" * 60)
    print("Quick Task Example - 단일 Agent 작업")
    print("=" * 60)
    print()

    orchestrator = await setup_orchestrator()

    # 단일 단계 작업
    task = {
        "name": "코드 설명",
        "query": "Python의 데코레이터가 무엇인지 간단히 설명해줘",
    }

    stages = [
        ConversationStage(
            name="explain",
            description="코드 설명",
            agent_capability="code_explanation",
        ),
    ]

    print(f"질문: {task['query']}")
    print("-" * 60)

    try:
        result = await orchestrator.execute(
            task=task,
            pattern=ConversationPattern.SEQUENTIAL,
            stages=stages,
            timeout_seconds=60,
        )

        print()
        print("답변:")
        if result.output:
            print(result.output.get("result", result.output))

    except Exception as e:
        print(f"오류 발생: {e}")


async def main():
    """메인 함수 - 예제 선택 및 실행."""
    print()
    print("Agent Orchestrator - Simple Pipeline Example")
    print("=" * 60)
    print()
    print("실행할 예제를 선택하세요:")
    print("1. 전체 파이프라인 (Research -> Code -> Review)")
    print("2. 단일 Agent 작업 (Quick Task)")
    print("3. 모든 예제 실행")
    print()

    choice = input("선택 (1/2/3, 기본값: 1): ").strip() or "1"

    if choice == "1":
        await run_simple_pipeline()
    elif choice == "2":
        await run_quick_task()
    elif choice == "3":
        await run_simple_pipeline()
        print("\n" + "=" * 60 + "\n")
        await run_quick_task()
    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    asyncio.run(main())
