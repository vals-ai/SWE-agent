from __future__ import annotations

import copy
import time

import together
from together.error import RateLimitError, InvalidRequestError
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from sweagent.agent.models import (
    GenericAPIModelConfig,
    InstanceStats,
    GLOBAL_STATS,
    AbstractModel,
    GLOBAL_STATS_LOCK,
)
from sweagent.exceptions import (
    ContentPolicyViolationError,
    ContextWindowExceededError,
    CostLimitExceededError,
    InstanceCallLimitExceededError,
    ModelConfigurationError,
)
from sweagent.tools.tools import ToolConfig
from sweagent.types import History, HistoryItem
from sweagent.utils.log import get_logger
import os

import dotenv

dotenv.load_dotenv()


MILLION = 1000000


class TogetherModel(AbstractModel):
    def __init__(self, args: GenericAPIModelConfig, tools: ToolConfig):
        self.config: GenericAPIModelConfig = args.model_copy(deep=True)
        self.stats = InstanceStats()
        self.tools = tools
        self.logger = get_logger("swea-lm", emoji="🤖")

        api_key = os.getenv("TOGETHER_API_KEY")

        self.logger.info(f"API Key present: {'✅' if api_key else '❌'}")

        self.client = together.Client(api_key=api_key)

        if self.config.max_input_tokens is not None:
            self.model_max_input_tokens = self.config.max_input_tokens
        else:
            msg = f"No max input tokens found for model {self.config.name!r}."

            self.logger.error(msg)
            raise ModelConfigurationError(msg)

        if self.config.max_output_tokens is not None:
            self.model_max_output_tokens = self.config.max_output_tokens
        else:
            msg = f"No max output tokens found for model {self.config.name!r}."

            self.logger.error(msg)
            raise ModelConfigurationError(msg)

    def _update_stats(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cost: float = 0.0,
        tool_names: list[str] | None = None,
    ) -> None:
        with GLOBAL_STATS_LOCK:
            GLOBAL_STATS.total_cost += cost

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

    def _single_query(
        self,
        messages: list[dict[str, str]],
        n: int | None = None,
        temperature: float | None = None,
    ) -> list[dict]:
        self._sleep()

        # TODO: Fix this later and get the actual input tokens
        # input_tokens = 0

        # if self.model_max_input_tokens and input_tokens > self.model_max_input_tokens:
        #     msg = f"Input tokens {input_tokens} exceed max tokens {self.model_max_input_tokens}"
        #     raise ContextWindowExceededError(msg)

        kwargs = {
            "model": self.config.name,
            "messages": messages,
            "temperature": self.config.temperature
            if temperature is None
            else temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.model_max_output_tokens,
            **self.config.completion_kwargs,
        }

        if self.tools.use_function_calling and self.tools.tools:
            kwargs["tools"] = self.tools.tools

        if n is not None:
            kwargs["n"] = n

        try:
            response = self.client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            raise RuntimeError(f"Rate limit exceeded: {e}") from e
        except InvalidRequestError as e:
            if "context length" in str(e).lower():
                raise ContextWindowExceededError from e
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error while process response {e}") from e

        self.logger.info(f"Model response: {response}")

        outputs = []
        output_tokens = response.usage.completion_tokens
        input_tokens = response.usage.prompt_tokens
        tool_names = None

        n_choices = n if n is not None else 1
        for i in range(min(n_choices, len(response.choices))):
            choice = response.choices[i]
            output = choice.message.content or ""
            output_dict = {"message": output}

            if self.tools.use_function_calling:
                if choice.message.tool_calls:
                    tool_calls = [
                        {
                            "type": call.type,
                            "id": call.id,
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in choice.message.tool_calls
                    ]
                    tool_names = [call["function"]["name"] for call in tool_calls]
                    output_dict["tool_calls"] = tool_calls
                else:
                    output_dict["tool_calls"] = []

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

    def _single_query_streaming(
        self,
        messages: list[dict[str, str]],
        n: int | None = None,
        temperature: float | None = None,
    ) -> list[dict]:
        self._sleep()

        kwargs = {
            "model": self.config.name,
            "messages": messages,
            "temperature": self.config.temperature
            if temperature is None
            else temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.model_max_output_tokens,
            "stream": True,
            **self.config.completion_kwargs,
        }

        if self.tools.use_function_calling and self.tools.tools:
            kwargs["tools"] = self.tools.tools
        try:
            stream = self.client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            raise RuntimeError(f"Rate limit exceeded: {e}") from e
        except InvalidRequestError as e:
            if "context length" in str(e).lower():
                raise ContextWindowExceededError from e
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error while process response {e}") from e

        complete_content = ""
        tool_calls = []
        output_tokens = 0
        input_tokens = 0

        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    complete_content += delta.content
                if delta.tool_calls:
                    tool_calls.extend(delta.tool_calls)

            if chunk.usage:
                output_tokens = chunk.usage.completion_tokens
                input_tokens = chunk.usage.prompt_tokens

        outputs = [{"message": complete_content}]
        tool_names = None

        if self.tools.use_function_calling and tool_calls:
            formatted_tool_calls = [
                {
                    "type": call.type,
                    "id": call.id,
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ]
            tool_names = [call["function"]["name"] for call in formatted_tool_calls]
            outputs[0]["tool_calls"] = formatted_tool_calls

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
        # If specified we will use the streaming api
        if self.config.completion_kwargs.get("stream", False):
            if n is None:
                return self._single_query_streaming(messages, temperature=temperature)
            outputs = []
            for _ in range(n):
                outputs.extend(
                    self._single_query_streaming(messages, temperature=temperature)
                )
            return outputs

        # Default is going to use the non-streaming api
        if n is None:
            return self._single_query(messages, temperature=temperature)
        outputs = []
        for _ in range(n):
            outputs.extend(self._single_query(messages, temperature=temperature))
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
                f"Retrying LM query: attempt {retry_state.attempt_number} "
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

    def _history_to_messages(self, history: History) -> list[dict[str, str]]:
        history = copy.deepcopy(history)

        def get_role(history_item: HistoryItem) -> str:
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
            messages.append(message)

        return messages
