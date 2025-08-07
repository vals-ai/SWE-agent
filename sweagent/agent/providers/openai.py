from __future__ import annotations

import time
from typing import Any
from openai import OpenAI
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from sweagent.exceptions import (
    ContentPolicyViolationError,
    ContextWindowExceededError,
    CostLimitExceededError,
    InstanceCallLimitExceededError,
    ModelConfigurationError,
)
from sweagent.tools.tools import ToolConfig
from sweagent.types import History
from sweagent.utils.log import get_logger
from sweagent.agent.models import (
    AbstractModel,
    GenericAPIModelConfig,
    InstanceStats,
    GLOBAL_STATS,
    GLOBAL_STATS_LOCK,
)
import copy

MILLION = 1000000


class OpenAIModel(AbstractModel):
    def __init__(self, args: GenericAPIModelConfig, tools: ToolConfig):
        """Model served by the OpenAI client."""
        # Always copy config to avoid shared state between different instances
        self.config: GenericAPIModelConfig = args.model_copy(deep=True)
        self.stats = InstanceStats()
        self.tools = tools
        self.logger = get_logger("swea-openai", emoji="🤖")

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.config.choose_api_key(),
            base_url=self.config.api_base if self.config.api_base else None,
        )

        if self.config.max_input_tokens is not None:
            self.model_max_input_tokens = self.config.max_input_tokens
        else:
            # Default values for OpenAI models
            self.model_max_input_tokens = 128000

        if self.config.max_output_tokens is not None:
            self.model_max_output_tokens = self.config.max_output_tokens
        else:
            # Default values for OpenAI models
            self.model_max_output_tokens = 4096

    @property
    def instance_cost_limit(self) -> float:
        """Cost limit for the model. Returns 0 if there is no limit."""
        return self.config.per_instance_cost_limit

    def _update_stats(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        tool_names: list[str] | None = None,
    ) -> None:
        with GLOBAL_STATS_LOCK:
            GLOBAL_STATS.total_cost += cost

        self.logger.info(
            f"==========\nInput tokens: {input_tokens}\nOutput tokens: {output_tokens}\nCost: {cost}\n=========="
        )

        self.stats.instance_cost += cost
        self.stats.tokens_sent += input_tokens
        self.stats.tokens_received += output_tokens
        self.stats.api_calls += 1

        if tool_names is not None:
            self.stats.update_tool_call_definition(tool_names)

        if 0 < self.config.per_instance_call_limit < self.stats.api_calls:
            msg = f"API calls {self.stats.api_calls} exceeds limit {self.config.per_instance_call_limit}"
            self.logger.warning(msg)
            raise InstanceCallLimitExceededError(msg)

    def _sleep(self) -> None:
        elapsed_time = time.time() - GLOBAL_STATS.last_query_timestamp
        if elapsed_time < self.config.delay:
            time.sleep(self.config.delay - elapsed_time)
        with GLOBAL_STATS_LOCK:
            GLOBAL_STATS.last_query_timestamp = time.time()

    def _validate_and_format_tools(self) -> list[dict] | None:
        """Validate and format tools for OpenAI Chat Completions API.

        Returns:
            List of valid tools in the format expected by OpenAI Chat Completions API, or None if no valid tools.
        """
        if not self.tools.use_function_calling or not self.tools.tools:
            return None

        valid_tools = []
        for tool in self.tools.tools:
            if (
                isinstance(tool, dict)
                and "function" in tool
                and isinstance(tool["function"], dict)
            ):
                function_data = tool["function"]
                if "name" in function_data:
                    # Format for Chat Completions API
                    formatted_tool = {
                        "type": "function",
                        "function": {
                            "name": function_data["name"],
                            "description": function_data.get("description", ""),
                            "parameters": function_data.get("parameters", {}),
                        },
                    }
                    valid_tools.append(formatted_tool)
                else:
                    self.logger.warning(f"Skipping tool without function.name: {tool}")
            else:
                self.logger.warning(f"Skipping invalid tool format: {tool}")

        if valid_tools:
            return valid_tools
        else:
            self.logger.warning(
                "No valid tools found, proceeding without function calling"
            )
            return None

    def _single_query_streaming(
        self,
        messages: list[dict[str, Any]],
        n: int | None = None,
        temperature: float | None = None,
    ) -> list[dict]:
        """Handle streaming responses using Chat Completions API"""
        self._sleep()

        request_params = {
            "model": self.config.name,
            "messages": messages,
            "stream": True,
        }

        valid_tools = self._validate_and_format_tools()
        if valid_tools:
            request_params["tools"] = valid_tools

        if temperature is not None:
            request_params["temperature"] = temperature
        elif self.config.temperature is not None:
            request_params["temperature"] = self.config.temperature

        if self.config.top_p is not None:
            request_params["top_p"] = self.config.top_p

        if self.config.max_output_tokens:
            request_params["max_completion_tokens"] = self.config.max_output_tokens

        if n is not None:
            request_params["n"] = n

        try:
            response_stream = self.client.chat.completions.create(**request_params)
        except Exception as e:
            if "context_length" in str(e).lower():
                raise ContextWindowExceededError from e
            elif "content_policy" in str(e).lower():
                raise ContentPolicyViolationError from e
            elif "cost" in str(e).lower():
                raise CostLimitExceededError from e
            else:
                raise

        complete_content = ""
        tool_calls = []
        tool_names = []
        input_tokens = 0
        output_tokens = 0

        for chunk in response_stream:
            if chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]

                # Handle content delta
                if choice.delta.content:
                    complete_content += choice.delta.content

                # Handle tool calls
                if choice.delta.tool_calls:
                    for tool_call in choice.delta.tool_calls:
                        # Extend tool_calls list if needed
                        while len(tool_calls) <= tool_call.index:
                            tool_calls.append(
                                {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            )

                        # Update tool call data
                        if tool_call.id:
                            tool_calls[tool_call.index]["id"] = tool_call.id
                        if tool_call.function:
                            if tool_call.function.name:
                                tool_calls[tool_call.index]["function"]["name"] = (
                                    tool_call.function.name
                                )
                                if tool_call.function.name not in tool_names:
                                    tool_names.append(tool_call.function.name)
                            if tool_call.function.arguments:
                                tool_calls[tool_call.index]["function"][
                                    "arguments"
                                ] += tool_call.function.arguments

            # Get usage info from the final chunk
            if hasattr(chunk, "usage") and chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

        self.logger.info(f"Model response: {complete_content}")

        outputs = [{"message": complete_content}]

        if tool_calls:
            # Filter out incomplete tool calls
            complete_tool_calls = [
                tc for tc in tool_calls if tc["id"] and tc["function"]["name"]
            ]
            outputs[0]["tool_calls"] = complete_tool_calls
            self.logger.info(f"Tool calls: {complete_tool_calls}")

        input_cost_for_turn = (self.config.input_cost * input_tokens) / MILLION
        output_cost_for_turn = (self.config.output_cost * output_tokens) / MILLION
        total_cost_for_turn = input_cost_for_turn + output_cost_for_turn

        self._update_stats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=total_cost_for_turn,
            tool_names=tool_names if tool_names else None,
        )

        return outputs

    def _single_query(
        self,
        messages: list[dict[str, Any]],
        n: int | None = None,
        temperature: float | None = None,
    ) -> list[dict]:
        """Handle non-streaming responses using Chat Completions API"""
        self._sleep()

        request_params = {
            "model": self.config.name,
            "messages": messages,
        }

        valid_tools = self._validate_and_format_tools()
        if valid_tools:
            request_params["tools"] = valid_tools

        if temperature is not None:
            request_params["temperature"] = temperature
        elif self.config.temperature is not None:
            request_params["temperature"] = self.config.temperature

        if self.config.top_p is not None:
            request_params["top_p"] = self.config.top_p

        if self.config.max_output_tokens:
            request_params["max_completion_tokens"] = self.config.max_output_tokens

        if n is not None:
            request_params["n"] = n

        if self.config.completion_kwargs:
            request_params.update(self.config.completion_kwargs)

        try:
            response = self.client.chat.completions.create(**request_params)
        except Exception as e:
            if "context_length" in str(e).lower():
                raise ContextWindowExceededError from e
            elif "content_policy" in str(e).lower():
                raise ContentPolicyViolationError from e
            elif "cost" in str(e).lower():
                raise CostLimitExceededError from e
            else:
                raise

        self.logger.info(f"Response: {response}")

        n_choices = n if n is not None else 1
        outputs = []
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        tool_names = None

        for i in range(n_choices):
            choice = response.choices[i]
            output = choice.message.content or ""
            output_dict = {"message": output}

            if self.tools.use_function_calling and choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in choice.message.tool_calls
                ]
                tool_names = [call["function"]["name"] for call in tool_calls]
                output_dict["tool_calls"] = tool_calls

            outputs.append(output_dict)

        input_cost_for_turn = (self.config.input_cost * input_tokens) / MILLION
        output_cost_for_turn = (self.config.output_cost * output_tokens) / MILLION
        total_cost_for_turn = input_cost_for_turn + output_cost_for_turn

        self._update_stats(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=total_cost_for_turn,
            tool_names=tool_names,
        )

        return outputs

    def _query(
        self,
        messages: list[dict[str, str]],
        n: int | None = None,
        temperature: float | None = None,
    ) -> list[dict]:
        if self.config.streaming:
            self.logger.info(f"Using streaming for {self.config.name}")
            if n is None:
                return self._single_query_streaming(messages, temperature=temperature)
            outputs = []
            for _ in range(n):
                outputs.extend(self._single_query_streaming(messages))
            return outputs
        else:
            self.logger.info(f"Using non-streaming for {self.config.name}")
            if n is None:
                return self._single_query(messages, temperature=temperature)
            outputs = []
            for _ in range(n):
                outputs.extend(self._single_query(messages))
            return outputs

    def query(
        self, history: History, n: int = 1, temperature: float | None = None
    ) -> list[dict] | dict:
        messages = self._history_to_messages(history)

        def retry_warning(retry_state: RetryCallState):
            exception_info = ""
            if (
                retry_state.outcome is not None
                and retry_state.outcome.exception() is not None
            ):
                exception = retry_state.outcome.exception()
                exception_info = (
                    f" due to {exception.__class__.__name__}: {str(exception)}"
                )

            self.logger.warning(
                f"Retrying OpenAI query: attempt {retry_state.attempt_number} "
                f"(slept for {retry_state.idle_for:.2f}s)"
                f"{exception_info}"
            )

        for attempt in Retrying(
            stop=stop_after_attempt(self.config.retry.retries),
            wait=wait_random_exponential(
                min=self.config.retry.min_wait, max=self.config.retry.max_wait
            ),
            reraise=True,
            retry=retry_if_not_exception_type(
                (
                    ContextWindowExceededError,
                    CostLimitExceededError,
                    RuntimeError,
                    ContentPolicyViolationError,
                    ModelConfigurationError,
                )
            ),
            before_sleep=retry_warning,
        ):
            with attempt:
                result = self._query(messages, n=n, temperature=temperature)
        if n is None or n == 1:
            return result[0]

        return result

    def _history_to_messages(
        self,
        history: History,
    ) -> list[dict[str, Any]]:
        """Format history for Chat Completions API"""
        history = copy.deepcopy(history)

        def get_role(history_item: dict[str, Any]) -> str:
            if history_item["role"] == "system":
                return "user" if self.config.convert_system_to_user else "system"
            return history_item["role"]

        messages = []
        for history_item in history:
            role = get_role(history_item)

            if role == "tool":
                message = {
                    "role": role,
                    "content": history_item["content"],
                    # Only one tool call per observation
                    "tool_call_id": history_item["tool_call_ids"][0],  # type: ignore
                }
            elif (tool_calls := history_item.get("tool_calls")) is not None:
                message = {
                    "role": role,
                    "content": history_item["content"],
                    "tool_calls": tool_calls,
                }
            else:
                message = {"role": role, "content": history_item["content"]}

            if "cache_control" in history_item:
                message["cache_control"] = history_item["cache_control"]

            messages.append(message)

        return messages
