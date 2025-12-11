"""Upstage API usage examples.

This module demonstrates how to use the Upstage LLM provider
for chat completions and document OCR.
"""

import asyncio
import os
from pathlib import Path

# Upstage API key 설정
# 환경 변수 또는 직접 설정
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY", "up_jlvcOFTtC4cUt0RKQ78oYQYHSZsKG")


async def chat_example():
    """Upstage Solar를 사용한 채팅 예제."""
    from src.llm import UpstageProvider

    # Provider 생성
    provider = UpstageProvider(api_key=UPSTAGE_API_KEY)

    print("=== Upstage Solar Chat Example ===\n")

    # 일반 채팅
    response = await provider.chat(
        messages=[{"role": "user", "content": "안녕하세요! 오늘 날씨가 어때요?"}],
        model="solar-pro2",
        max_tokens=1024,
        temperature=0.7,
    )

    print(f"Model: {response.model}")
    print(f"Response: {response.content}")
    print(f"Usage: {response.usage}")
    print()


async def chat_stream_example():
    """Upstage Solar 스트리밍 채팅 예제."""
    from src.llm import UpstageProvider

    provider = UpstageProvider(api_key=UPSTAGE_API_KEY)

    print("=== Upstage Solar Streaming Example ===\n")

    print("Response: ", end="")
    async for chunk in provider.chat_stream(
        messages=[{"role": "user", "content": "파이썬에 대해 간단히 설명해주세요."}],
        model="solar-pro2",
        max_tokens=512,
    ):
        print(chunk, end="", flush=True)
    print("\n")


async def chat_with_system_prompt_example():
    """시스템 프롬프트를 사용한 채팅 예제."""
    from src.llm import UpstageProvider

    provider = UpstageProvider(api_key=UPSTAGE_API_KEY)

    print("=== Upstage Solar with System Prompt ===\n")

    response = await provider.chat(
        messages=[{"role": "user", "content": "What is machine learning?"}],
        model="solar-pro2",
        system_prompt="You are a helpful AI assistant that explains technical concepts in simple terms. Answer in Korean.",
        max_tokens=1024,
    )

    print(f"Response: {response.content}")
    print()


async def ocr_example(file_path: str):
    """Upstage Document Digitization (OCR) 예제."""
    from src.llm import UpstageOCR

    ocr = UpstageOCR(api_key=UPSTAGE_API_KEY)

    print("=== Upstage Document OCR Example ===\n")

    # 파일이 존재하는지 확인
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        print("Please provide a valid document file path.")
        return

    # OCR 수행
    result = await ocr.digitize(
        file_path=file_path,
        model="ocr",
    )

    print(f"OCR Result: {result}")
    print()


async def document_parse_example(file_path: str):
    """Upstage Document Parsing 예제 (복잡한 문서용)."""
    from src.llm import UpstageOCR

    ocr = UpstageOCR(api_key=UPSTAGE_API_KEY)

    print("=== Upstage Document Parsing Example ===\n")

    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return

    # Document parse (표, 이미지 등이 포함된 복잡한 문서)
    result = await ocr.parse_document(
        file_path=file_path,
        output_format="markdown",
    )

    print(f"Parsed Document: {result}")
    print()


def factory_example():
    """LLMProviderFactory를 사용한 Provider 생성 예제."""
    from src.llm import LLMProviderFactory, get_provider

    print("=== LLM Provider Factory Example ===\n")

    # 방법 1: Provider 이름으로 생성
    os.environ["UPSTAGE_API_KEY"] = UPSTAGE_API_KEY
    upstage_provider = LLMProviderFactory.create(provider="upstage")
    print(f"Created provider: {upstage_provider.provider_name}")
    print(f"Default model: {upstage_provider.default_model}")

    # 방법 2: 모델 이름으로 자동 Provider 선택
    provider = get_provider(model="solar-pro2")
    print(f"Auto-selected provider for solar-pro2: {provider.provider_name}")

    # 사용 가능한 Provider 목록
    print(f"\nAvailable providers: {LLMProviderFactory.list_providers()}")

    # 사용 가능한 모델 목록
    print(f"Upstage models: {LLMProviderFactory.list_models('upstage')}")
    print()


async def agent_with_upstage_example():
    """Agent에서 Upstage 사용 예제."""
    from src.llm import UpstageProvider
    from src.agents.base import SimpleAgent
    from src.models import AgentConfig, AgentCapability, Message

    print("=== Agent with Upstage Provider Example ===\n")

    # Upstage를 사용하는 Agent 설정
    config = AgentConfig(
        agent_id="upstage_agent",
        name="Upstage Solar Agent",
        description="An agent powered by Upstage Solar",
        model="solar-pro2",  # Upstage 모델 사용
        capabilities=[
            AgentCapability(
                name="general",
                description="General conversation",
            )
        ],
        system_prompt="You are a helpful assistant powered by Upstage Solar. Answer in Korean.",
    )

    # Agent 생성 및 Upstage Provider 설정
    agent = SimpleAgent(config)
    agent.set_llm_provider(UpstageProvider(api_key=UPSTAGE_API_KEY))

    # 메시지 처리
    message = Message.create_task(
        sender_id="user",
        recipient_id="upstage_agent",
        content={"task": "인공지능의 미래에 대해 간단히 설명해주세요."},
    )

    response = await agent.process(message)
    print(f"Agent Response: {response.content.get('result', '')}")
    print()


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Upstage API Examples")
    print("=" * 60)
    print()

    # Factory 예제 (동기)
    factory_example()

    # 채팅 예제들
    await chat_example()
    await chat_stream_example()
    await chat_with_system_prompt_example()

    # Agent 예제
    await agent_with_upstage_example()

    # OCR 예제 (파일 경로가 필요함)
    # await ocr_example("path/to/your/document.pdf")
    # await document_parse_example("path/to/your/complex-document.pdf")


if __name__ == "__main__":
    asyncio.run(main())
