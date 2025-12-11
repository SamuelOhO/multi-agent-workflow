"""Coder Agent - Specialized agent for code generation and modification.

This agent handles code-related tasks including generation, modification,
explanation, and debugging.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent, LLMError
from src.models import AgentConfig, Message


class CoderAgent(BaseAgent):
    """Agent specialized for code generation and manipulation.

    Capabilities:
    - code_generation: Generate new code from requirements
    - code_modification: Modify existing code
    - code_explanation: Explain code functionality
    - debugging: Help debug code issues

    The agent produces clean, well-documented code following
    best practices for the specified programming language.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the coder agent.

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
        """Get the default system prompt for the coder agent."""
        return """You are an expert Software Engineer Agent specialized in writing clean, efficient, and well-documented code.

Your responsibilities:
1. Generate high-quality code based on requirements
2. Follow best practices and design patterns
3. Write clear comments and documentation
4. Handle edge cases and errors appropriately
5. Optimize for readability and maintainability

Code Standards:
- Use meaningful variable and function names
- Follow language-specific conventions and style guides
- Include type hints where applicable
- Write modular, reusable code
- Add inline comments for complex logic

Output Format:
- Always wrap code in appropriate markdown code blocks with language specification
- Provide a brief explanation of the implementation
- List any dependencies or requirements
- Include usage examples when helpful
- Note any assumptions made"""

    async def process(self, message: Message) -> Message:
        """Process a coding task message.

        Handles:
        - code_generation: Generate new code
        - code_modification: Modify existing code
        - code_explanation: Explain code
        - debugging: Debug code issues

        Args:
            message: The incoming message containing the coding task.

        Returns:
            A response message with code output.
        """
        try:
            self.set_busy()

            content = message.content
            capability = content.get("capability", "code_generation")

            if capability == "code_generation":
                result = await self._handle_code_generation(content)
            elif capability == "code_modification":
                result = await self._handle_code_modification(content)
            elif capability == "code_explanation":
                result = await self._handle_code_explanation(content)
            elif capability == "debugging":
                result = await self._handle_debugging(content)
            else:
                # Default to general coding task
                result = await self._handle_general_coding(content)

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
                error_code="CODER_LLM_ERROR",
            )
        except Exception as e:
            self.set_error()
            return self._create_error_response(
                original_message=message,
                error_message=f"Coding task failed: {e}",
                error_code="CODER_ERROR",
            )

    async def _handle_code_generation(self, content: dict[str, Any]) -> dict[str, Any]:
        """Handle code generation capability.

        Args:
            content: Message content containing generation requirements.

        Returns:
            Generated code dictionary.
        """
        requirements = content.get("requirements", content.get("task", ""))
        language = content.get("language", "python")
        style = content.get("style", "clean and readable")

        prompt = f"""Generate code based on the following requirements:

Requirements: {requirements}
Programming Language: {language}
Code Style: {style}

Please provide:
1. Complete, working code implementation
2. Clear comments explaining key parts
3. Any necessary imports or dependencies
4. Example usage if applicable
5. Notes on potential improvements or considerations

Generate the code:"""

        response = await self._call_llm(prompt)

        return {
            "code": response,
            "language": language,
            "capability": "code_generation",
            "requirements": requirements,
        }

    async def _handle_code_modification(
        self, content: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle code modification capability.

        Args:
            content: Message content containing code and modification instructions.

        Returns:
            Modified code dictionary.
        """
        original_code = content.get("code", "")
        modification = content.get("modification", content.get("task", ""))
        language = content.get("language", "python")

        prompt = f"""Modify the following code according to the instructions:

Original Code:
```{language}
{original_code}
```

Modification Instructions: {modification}

Please provide:
1. The modified code with changes applied
2. Explanation of what was changed and why
3. Any potential side effects or considerations
4. Suggestions for related improvements (if any)

Modified code:"""

        response = await self._call_llm(prompt)

        return {
            "modified_code": response,
            "original_code": original_code,
            "modification": modification,
            "language": language,
            "capability": "code_modification",
        }

    async def _handle_code_explanation(self, content: dict[str, Any]) -> dict[str, Any]:
        """Handle code explanation capability.

        Args:
            content: Message content containing code to explain.

        Returns:
            Code explanation dictionary.
        """
        code = content.get("code", content.get("task", ""))
        language = content.get("language", "python")
        detail_level = content.get("detail_level", "comprehensive")

        prompt = f"""Explain the following code in detail:

Code:
```{language}
{code}
```

Detail Level: {detail_level}

Please provide:
1. Overall purpose and functionality
2. Step-by-step explanation of the logic
3. Explanation of key functions/methods
4. Data flow description
5. Any notable patterns or techniques used
6. Potential issues or areas for improvement

Explain the code:"""

        response = await self._call_llm(prompt)

        return {
            "explanation": response,
            "code": code,
            "language": language,
            "capability": "code_explanation",
        }

    async def _handle_debugging(self, content: dict[str, Any]) -> dict[str, Any]:
        """Handle code debugging capability.

        Args:
            content: Message content containing code and error information.

        Returns:
            Debugging results dictionary.
        """
        code = content.get("code", "")
        error = content.get("error", content.get("task", ""))
        language = content.get("language", "python")

        prompt = f"""Debug the following code and fix the issue:

Code:
```{language}
{code}
```

Error/Problem: {error}

Please provide:
1. Analysis of the problem
2. Root cause identification
3. Fixed code with the bug resolved
4. Explanation of the fix
5. Suggestions to prevent similar issues

Debug and fix:"""

        response = await self._call_llm(prompt)

        return {
            "debug_result": response,
            "original_code": code,
            "reported_error": error,
            "language": language,
            "capability": "debugging",
        }

    async def _handle_general_coding(self, content: dict[str, Any]) -> dict[str, Any]:
        """Handle general coding tasks.

        Args:
            content: Message content containing coding task.

        Returns:
            Coding results dictionary.
        """
        task = content.get("task", str(content))
        language = content.get("language", "python")

        prompt = f"""Complete the following coding task:

Task: {task}
Preferred Language: {language}

Please provide a complete solution with:
1. Working code implementation
2. Clear documentation
3. Usage examples
4. Any relevant notes or considerations

Complete the task:"""

        response = await self._call_llm(prompt)

        return {
            "result": response,
            "task": task,
            "language": language,
            "capability": "general_coding",
        }
