"""Research Agent - Specialized agent for web search and information gathering.

This agent performs research tasks including web search simulation,
information extraction, and summarization.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent, LLMError
from src.models import AgentConfig, Message


class ResearchAgent(BaseAgent):
    """Agent specialized for research and information gathering.

    Capabilities:
    - web_search: Search the internet for information
    - summarization: Summarize text content
    - data_extraction: Extract structured data from text

    The agent uses the LLM to simulate search capabilities and
    provide comprehensive research outputs.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the research agent.

        Args:
            config: Agent configuration.
        """
        super().__init__(config)

        # Set default system prompt if not provided
        if not self._config.system_prompt:
            self._config = AgentConfig(
                **{
                    **self._config.model_dump(),
                    "system_prompt": self._get_default_system_prompt(),
                }
            )

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the research agent."""
        return """You are a professional Research Agent specialized in information gathering and analysis.

Your responsibilities:
1. Conduct thorough research on given topics
2. Extract key information and insights
3. Summarize findings in a clear, structured format
4. Cite sources when available
5. Highlight uncertain or unverified information

Output Guidelines:
- Always structure your response with clear sections
- Use bullet points for key findings
- Indicate confidence level for each finding
- Provide a brief summary at the beginning
- List sources or references at the end

When searching for information:
- Consider multiple perspectives
- Look for recent and reliable sources
- Cross-reference information when possible
- Note any conflicting information found"""

    async def process(self, message: Message) -> Message:
        """Process a research task message.

        Handles:
        - web_search: Search for information on a topic
        - summarization: Summarize provided text
        - data_extraction: Extract structured data

        Args:
            message: The incoming message containing the research task.

        Returns:
            A response message with research results.
        """
        try:
            self.set_busy()

            # Determine the type of research task
            content = message.content
            capability = content.get("capability", "web_search")

            if capability == "web_search":
                result = await self._handle_web_search(content)
            elif capability == "summarization":
                result = await self._handle_summarization(content)
            elif capability == "data_extraction":
                result = await self._handle_data_extraction(content)
            else:
                # Default to general research
                result = await self._handle_general_research(content)

            self.activate()

            return self._create_response(
                original_message=message,
                content=result,
            )

        except LLMError as e:
            self.set_error()
            return self._create_error_response(
                original_message=message,
                error_message=str(e),
                error_code="RESEARCH_LLM_ERROR",
            )
        except Exception as e:
            self.set_error()
            return self._create_error_response(
                original_message=message,
                error_message=f"Research failed: {e}",
                error_code="RESEARCH_ERROR",
            )

    async def _handle_web_search(self, content: dict[str, Any]) -> dict[str, Any]:
        """Handle web search capability.

        Args:
            content: Message content containing search parameters.

        Returns:
            Search results dictionary.
        """
        query = content.get("query", content.get("task", ""))
        max_results = content.get("max_results", 10)

        prompt = f"""Perform a web search for the following query and provide comprehensive results.

Query: {query}
Maximum Results: {max_results}

Please provide:
1. A summary of the main findings
2. Key facts and information discovered
3. Different perspectives or viewpoints if applicable
4. Any relevant statistics or data
5. Suggested related topics for further research

Format your response as structured research findings."""

        response = await self._call_llm(prompt)

        return {
            "result": response,
            "query": query,
            "capability": "web_search",
            "max_results": max_results,
        }

    async def _handle_summarization(self, content: dict[str, Any]) -> dict[str, Any]:
        """Handle text summarization capability.

        Args:
            content: Message content containing text to summarize.

        Returns:
            Summary results dictionary.
        """
        text = content.get("text", content.get("task", ""))
        max_length = content.get("max_length", 500)

        prompt = f"""Summarize the following text concisely while preserving key information.

Text to summarize:
{text}

Requirements:
- Maximum summary length: approximately {max_length} characters
- Preserve the main ideas and key points
- Maintain logical flow
- Use clear and concise language

Provide the summary:"""

        response = await self._call_llm(prompt)

        return {
            "summary": response,
            "original_length": len(text),
            "capability": "summarization",
        }

    async def _handle_data_extraction(self, content: dict[str, Any]) -> dict[str, Any]:
        """Handle structured data extraction capability.

        Args:
            content: Message content containing text for extraction.

        Returns:
            Extracted data dictionary.
        """
        text = content.get("text", content.get("task", ""))
        extract_fields = content.get("fields", [])

        fields_str = (
            ", ".join(extract_fields)
            if extract_fields
            else "all relevant entities and facts"
        )

        prompt = f"""Extract structured data from the following text.

Text:
{text}

Fields to extract: {fields_str}

Please extract and organize the information in a structured format:
- Identify key entities (people, organizations, locations, dates)
- Extract relevant facts and figures
- Note relationships between entities
- Highlight any quantitative data

Provide the extracted data in a clear, organized format."""

        response = await self._call_llm(prompt)

        return {
            "extracted_data": response,
            "fields_requested": extract_fields,
            "capability": "data_extraction",
        }

    async def _handle_general_research(self, content: dict[str, Any]) -> dict[str, Any]:
        """Handle general research tasks.

        Args:
            content: Message content containing research task.

        Returns:
            Research results dictionary.
        """
        task = content.get("task", content.get("query", str(content)))

        prompt = f"""Conduct comprehensive research on the following topic:

Topic/Task: {task}

Please provide:
1. Overview and background
2. Key findings and facts
3. Different perspectives or approaches
4. Recent developments (if applicable)
5. Recommendations or conclusions
6. Suggested areas for further investigation

Structure your response clearly with sections and bullet points."""

        response = await self._call_llm(prompt)

        return {
            "result": response,
            "task": task,
            "capability": "general_research",
        }
