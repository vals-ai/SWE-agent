from __future__ import annotations

import copy
import time
import litellm
import litellm.types.utils
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
from sweagent.types import History, HistoryItem
from sweagent.utils.log import get_logger
from sweagent.agent.models import (
    AbstractModel,
    GenericAPIModelConfig,
    InstanceStats,
    GLOBAL_STATS,
    GLOBAL_STATS_LOCK,
)


litellm.suppress_debug_info = True
litellm.modify_params = True
litellm.drop_params = True

MILLION = 1000000


class LiteLLMModel(AbstractModel):
    def __init__(self, args: GenericAPIModelConfig, tools: ToolConfig):
        """Model served by the `litellm` library."""
        # Always copy config to avoid shared state between different instances
        self.config: GenericAPIModelConfig = args.model_copy(deep=True)
        self.stats = InstanceStats()
        self.tools = tools
        self.logger = get_logger("swea-lm", emoji="🤖")

        if self.config.max_input_tokens is not None:
            self.model_max_input_tokens = self.config.max_input_tokens
        else:
            self.model_max_input_tokens = litellm.model_cost.get(
                self.config.name, {}
            ).get("max_input_tokens")

        if self.config.max_output_tokens is not None:
            self.model_max_output_tokens = self.config.max_output_tokens
        else:
            self.model_max_output_tokens = litellm.model_cost.get(
                self.config.name, {}
            ).get("max_output_tokens")

        self.lm_provider = litellm.model_cost.get(self.config.name, {}).get(
            "litellm_provider"
        )

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

    def _single_query(
        self,
        messages: list[dict[str, str]],
        n: int | None = None,
        temperature: float | None = None,
    ) -> list[dict]:
        self._sleep()

        extra_args = {}
        if self.config.api_base:
            extra_args["api_base"] = self.config.api_base

        if self.tools.use_function_calling:
            extra_args["tools"] = self.tools.tools

        completion_kwargs = self.config.completion_kwargs
        if self.lm_provider == "anthropic":
            completion_kwargs["max_tokens"] = self.model_max_output_tokens

        try:
            response: litellm.types.utils.ModelResponse = litellm.completion(  # type: ignore
                model=self.config.name,
                messages=messages,
                temperature=(
                    self.config.temperature if temperature is None else temperature
                ),
                top_p=self.config.top_p,
                api_version=self.config.api_version,
                api_key=self.config.choose_api_key(),
                fallbacks=self.config.fallbacks,
                **completion_kwargs,
                **extra_args,
                n=n,
            )
        except litellm.exceptions.ContextWindowExceededError as e:
            raise ContextWindowExceededError from e
        except litellm.exceptions.ContentPolicyViolationError as e:
            raise ContentPolicyViolationError from e
        except litellm.exceptions.BadRequestError as e:
            if "is longer than the model's context length" in str(e):
                raise ContextWindowExceededError from e
            raise

        self.logger.info(f"Response: {response}")

        choices: litellm.types.utils.Choices = response.choices  # type: ignore

        n_choices = n if n is not None else 1

        outputs = []
        output_tokens = response.usage.completion_tokens
        input_tokens = response.usage.prompt_tokens
        tool_names = None

        for i in range(n_choices):
            output = choices[i].message.content or ""
            output_dict = {"message": output}

            if self.tools.use_function_calling:
                if response.choices[i].message.tool_calls:  # type: ignore
                    tool_calls = [
                        call.to_dict()
                        for call in response.choices[i].message.tool_calls
                    ]  # type: ignore
                    tool_names = [call["function"]["name"] for call in tool_calls]
                else:
                    tool_calls = []

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

    def _single_query_streaming(
        self,
        messages: list[dict[str, str]],
        n: int | None = None,
        temperature: float | None = None,
    ) -> list[dict]:
        """Handle streaming-only models by collecting chunks and building complete response"""
        self._sleep()

        extra_args = {}
        if self.config.api_base:
            extra_args["api_base"] = self.config.api_base
        if self.tools.use_function_calling:
            extra_args["tools"] = self.tools.tools

        completion_kwargs = self.config.completion_kwargs

        try:
            response = litellm.completion(
                model=self.config.name,
                messages=messages,
                temperature=(
                    self.config.temperature if temperature is None else temperature
                ),
                top_p=self.config.top_p,
                api_version=self.config.api_version,
                api_key=self.config.choose_api_key(),
                fallbacks=self.config.fallbacks,
                stream=True,
                **completion_kwargs,
                **extra_args,
                n=n,
            )
        except litellm.exceptions.ContextWindowExceededError as e:
            raise ContextWindowExceededError from e
        except litellm.exceptions.ContentPolicyViolationError as e:
            raise ContentPolicyViolationError from e
        except litellm.exceptions.BadRequestError as e:
            if "is longer than the model's context length" in str(e):
                raise ContextWindowExceededError from e
            raise

        chunks = []
        complete_content = ""
        for chunk in response:
            chunks.append(chunk)
            if hasattr(chunk, "choices") and chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    complete_content += delta.content

        complete_response = litellm.stream_chunk_builder(chunks, messages=messages)

        self.logger.info(f"Response: {complete_response}")
        cost = 0
        outputs = []
        output_tokens = 0
        tool_names = None

        if complete_response:
            output_tokens = (
                complete_response.usage.completion_tokens
                + complete_response.usage.completion_tokens_details.reasoning_tokens
            )

        outputs = [{"message": complete_content}]

        try:
            tool_calls = [
                call.to_dict()
                for call in complete_response.choices[0].message.tool_calls
            ]
            tool_names = [call["function"]["name"] for call in tool_calls]
            outputs[0]["tool_calls"] = tool_calls
        except Exception as e:
            self.logger.debug(f"Error extracting tool calls: {e}")

        final_input_tokens = complete_response.usage.prompt_tokens

        self._update_stats(
            input_tokens=final_input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            tool_names=tool_names,
        )
        return outputs

    def _query(
        self,
        messages: list[dict[str, str]],
        n: int | None = None,
        temperature: float | None = None,
    ) -> list[dict]:
        # Use streaming for grok-4-0709 model which only supports streaming
        if self.config.streaming:
            self.logger.info(f"Using streaming for {self.config.name}")
            if n is None:
                return self._single_query_streaming(messages, temperature=temperature)
            outputs = []
            for _ in range(n):
                outputs.extend(self._single_query_streaming(messages))
            return outputs

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
                attempt.retry_state.outcome is not None
                and attempt.retry_state.outcome.exception() is not None
            ):
                exception = attempt.retry_state.outcome.exception()
                exception_info = (
                    f" due to {exception.__class__.__name__}: {str(exception)}"
                )

            self.logger.warning(
                f"Retrying LM query: attempt {attempt.retry_state.attempt_number} "
                f"(slept for {attempt.retry_state.idle_for:.2f}s)"
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
                    litellm.exceptions.UnsupportedParamsError,
                    litellm.exceptions.NotFoundError,
                    litellm.exceptions.PermissionDeniedError,
                    litellm.exceptions.ContextWindowExceededError,
                    litellm.exceptions.APIError,
                    litellm.exceptions.ContentPolicyViolationError,
                    TypeError,
                    litellm.exceptions.AuthenticationError,
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
    ) -> list[dict[str, str]]:
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
                    # Only one tool call per observations
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
