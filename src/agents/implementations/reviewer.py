"""Reviewer Agent - Specialized agent for code review and quality assurance.

This agent performs code reviews, identifies issues, and suggests improvements
following best practices and coding standards.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent, LLMError
from src.models import AgentConfig, Message


class ReviewerAgent(BaseAgent):
    """Agent specialized for code review and quality assurance.

    Capabilities:
    - code_review: Comprehensive code review
    - security_review: Security-focused review
    - performance_review: Performance analysis
    - style_review: Style and convention checking

    The agent provides detailed feedback with actionable suggestions
    for code improvement.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the reviewer agent.

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
        """Get the default system prompt for the reviewer agent."""
        return """You are an expert Code Reviewer Agent with deep knowledge of software engineering best practices.

Your responsibilities:
1. Thoroughly review code for quality, correctness, and maintainability
2. Identify bugs, security vulnerabilities, and performance issues
3. Suggest improvements following industry best practices
4. Ensure code follows style guidelines and conventions
5. Provide constructive, actionable feedback

Review Criteria:
- Correctness: Does the code work as intended?
- Security: Are there any security vulnerabilities?
- Performance: Are there any performance concerns?
- Readability: Is the code easy to understand?
- Maintainability: Is the code easy to modify and extend?
- Testing: Is the code testable? Are edge cases handled?
- Documentation: Is the code well-documented?

Feedback Format:
- Use severity levels: CRITICAL, MAJOR, MINOR, SUGGESTION
- Provide specific line references when possible
- Include code examples for suggested fixes
- Explain the reasoning behind each comment
- Summarize overall code quality"""

    async def process(self, message: Message) -> Message:
        """Process a code review task message.

        Handles:
        - code_review: General code review
        - security_review: Security-focused review
        - performance_review: Performance analysis
        - style_review: Style checking

        Args:
            message: The incoming message containing code to review.

        Returns:
            A response message with review results.
        """
        try:
            self.set_busy()

            content = message.content
            capability = content.get("capability", "code_review")

            if capability == "code_review":
                result = await self._handle_code_review(content)
            elif capability == "security_review":
                result = await self._handle_security_review(content)
            elif capability == "performance_review":
                result = await self._handle_performance_review(content)
            elif capability == "style_review":
                result = await self._handle_style_review(content)
            else:
                # Default to general code review
                result = await self._handle_code_review(content)

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
                error_code="REVIEWER_LLM_ERROR",
            )
        except Exception as e:
            self.set_error()
            return self._create_error_response(
                original_message=message,
                error_message=f"Review failed: {e}",
                error_code="REVIEWER_ERROR",
            )

    async def _handle_code_review(self, content: dict[str, Any]) -> dict[str, Any]:
        """Handle comprehensive code review capability.

        Args:
            content: Message content containing code to review.

        Returns:
            Review results dictionary.
        """
        code = content.get("code", content.get("task", ""))
        language = content.get("language", "python")
        context = content.get("context", "")

        context_section = f"\nContext: {context}" if context else ""

        prompt = f"""Perform a comprehensive code review on the following code:

Language: {language}{context_section}

Code:
```{language}
{code}
```

Please provide a detailed review including:

1. **Summary**: Overall assessment of code quality (score 1-10)

2. **Critical Issues**: Bugs, errors that will cause failures
   - [CRITICAL] Issue description with line reference
   - Suggested fix with code example

3. **Major Issues**: Significant problems that should be addressed
   - [MAJOR] Issue description
   - Recommendation

4. **Minor Issues**: Small improvements recommended
   - [MINOR] Issue description
   - Suggestion

5. **Suggestions**: Optional enhancements
   - [SUGGESTION] Enhancement idea
   - Benefit explanation

6. **Positive Aspects**: What's done well

7. **Overall Recommendations**: Summary of key actions needed

Review the code:"""

        response = await self._call_llm(prompt)

        return {
            "review": response,
            "code": code,
            "language": language,
            "capability": "code_review",
        }

    async def _handle_security_review(self, content: dict[str, Any]) -> dict[str, Any]:
        """Handle security-focused code review.

        Args:
            content: Message content containing code to review.

        Returns:
            Security review results dictionary.
        """
        code = content.get("code", content.get("task", ""))
        language = content.get("language", "python")

        prompt = f"""Perform a security-focused code review on the following code:

Language: {language}

Code:
```{language}
{code}
```

Focus on identifying:

1. **Critical Security Vulnerabilities**
   - SQL Injection
   - Cross-Site Scripting (XSS)
   - Command Injection
   - Path Traversal
   - Authentication/Authorization flaws

2. **Data Security Issues**
   - Sensitive data exposure
   - Insufficient encryption
   - Insecure data storage
   - Data validation problems

3. **Input Validation**
   - Missing or inadequate validation
   - Sanitization issues
   - Type checking problems

4. **Authentication & Authorization**
   - Weak authentication
   - Missing authorization checks
   - Session management issues

5. **Security Best Practices**
   - OWASP Top 10 compliance
   - Secure coding guidelines
   - Defense in depth

For each issue found:
- Severity level (CRITICAL/HIGH/MEDIUM/LOW)
- Description of the vulnerability
- Potential impact
- Remediation steps with code examples

Security review:"""

        response = await self._call_llm(prompt)

        return {
            "security_review": response,
            "code": code,
            "language": language,
            "capability": "security_review",
        }

    async def _handle_performance_review(
        self, content: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle performance-focused code review.

        Args:
            content: Message content containing code to review.

        Returns:
            Performance review results dictionary.
        """
        code = content.get("code", content.get("task", ""))
        language = content.get("language", "python")

        prompt = f"""Perform a performance analysis on the following code:

Language: {language}

Code:
```{language}
{code}
```

Analyze the following aspects:

1. **Time Complexity**
   - Big O analysis of key operations
   - Potential bottlenecks
   - Unnecessary computations

2. **Space Complexity**
   - Memory usage patterns
   - Potential memory leaks
   - Inefficient data structures

3. **Algorithm Efficiency**
   - Better algorithm alternatives
   - Optimization opportunities
   - Caching possibilities

4. **Resource Usage**
   - I/O operations efficiency
   - Database query optimization
   - Network call efficiency

5. **Concurrency Considerations**
   - Parallelization opportunities
   - Thread safety issues
   - Async/await usage

6. **Recommendations**
   - Priority-ranked improvements
   - Expected impact of each optimization
   - Code examples for optimizations

Performance analysis:"""

        response = await self._call_llm(prompt)

        return {
            "performance_review": response,
            "code": code,
            "language": language,
            "capability": "performance_review",
        }

    async def _handle_style_review(self, content: dict[str, Any]) -> dict[str, Any]:
        """Handle code style review.

        Args:
            content: Message content containing code to review.

        Returns:
            Style review results dictionary.
        """
        code = content.get("code", content.get("task", ""))
        language = content.get("language", "python")
        style_guide = content.get(
            "style_guide", "PEP 8" if language == "python" else "standard"
        )

        prompt = f"""Review the following code for style and conventions:

Language: {language}
Style Guide: {style_guide}

Code:
```{language}
{code}
```

Check for:

1. **Naming Conventions**
   - Variable names
   - Function/method names
   - Class names
   - Constants

2. **Code Formatting**
   - Indentation consistency
   - Line length
   - Whitespace usage
   - Bracket placement

3. **Documentation**
   - Docstrings/comments presence
   - Comment quality
   - Documentation completeness

4. **Code Organization**
   - Import ordering
   - Function/class ordering
   - Module structure

5. **Best Practices**
   - Language idioms
   - Code patterns
   - Anti-patterns to avoid

For each issue:
- Line reference if applicable
- Current code
- Suggested improvement
- Rationale

Style review:"""

        response = await self._call_llm(prompt)

        return {
            "style_review": response,
            "code": code,
            "language": language,
            "style_guide": style_guide,
            "capability": "style_review",
        }
