"""
Shared token processing helpers for text generation.

These stateless helpers are used by both single-request and batch generation paths
to ensure consistent behavior across both execution modes.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any, Callable

from pydantic import ValidationError

from exo.shared.types.worker.runner_response import ToolCallItem

if TYPE_CHECKING:
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    from exo.worker.engines.mlx import Model

# Lazy imports for optional dependencies
_openai_harmony_available: bool | None = None


def _check_openai_harmony() -> bool:
    """Check if openai_harmony is available (for GPT-OSS support)."""
    global _openai_harmony_available
    if _openai_harmony_available is None:
        try:
            import openai_harmony  # noqa: F401
            _openai_harmony_available = True
        except ImportError:
            _openai_harmony_available = False
    return _openai_harmony_available


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Kimi tool call section tokens to filter out
KIMI_FILTER_TOKENS = frozenset({
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
})


# -----------------------------------------------------------------------------
# Token Filter (model-specific, constructed once per request)
# -----------------------------------------------------------------------------

@dataclass
class TokenFilter:
    """
    Model-specific token filtering, constructed once per request.
    
    Encapsulates model-specific behavior (Kimi filtering, GLM think prepend)
    so callers don't need to track model type.
    """
    skip_tokens: frozenset[str] = field(default_factory=frozenset)
    first_token_prepend: str | None = None
    first_token_prepend_id: int = 0
    _prepend_emitted: bool = field(default=False, repr=False)
    
    @classmethod
    def for_model(
        cls,
        model_id: str,
        needs_think_prepend: bool,
        tokenizer,  # TokenizerWrapper
    ) -> "TokenFilter":
        """
        Create a TokenFilter for the given model.
        
        Args:
            model_id: The model ID string (e.g., "kimi-k2", "glm-4.7")
            needs_think_prepend: Whether the thinking tag was consumed by chat template
            tokenizer: The tokenizer (for think_start token)
        """
        skip = frozenset()
        prepend = None
        prepend_id = 0
        
        # Kimi models: filter out tool call section markers
        if "kimi" in model_id.lower():
            skip = KIMI_FILTER_TOKENS
        
        # GLM/other thinking models: prepend think tag that was consumed by template
        if needs_think_prepend:
            prepend = getattr(tokenizer, 'think_start', None)
            prepend_id = getattr(tokenizer, 'think_start_id', 0)
        
        return cls(
            skip_tokens=skip,
            first_token_prepend=prepend,
            first_token_prepend_id=prepend_id,
        )
    
    def should_skip(self, token_text: str) -> bool:
        """Check if this token should be filtered out."""
        return token_text in self.skip_tokens
    
    def get_prepend(self) -> tuple[str, int] | None:
        """
        Get the prepend text/id if not yet emitted.
        
        Returns (text, token_id) on first call, None thereafter.
        """
        if self.first_token_prepend and not self._prepend_emitted:
            self._prepend_emitted = True
            return (self.first_token_prepend, self.first_token_prepend_id)
        return None


# -----------------------------------------------------------------------------
# Token Processing State (for stateful processing across tokens)
# -----------------------------------------------------------------------------

@dataclass
class ToolCallState:
    """State for tool call accumulation across multiple tokens."""
    in_tool_call: bool = False
    tool_call_parts: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.in_tool_call = False
        self.tool_call_parts = []


@dataclass
class ThinkingState:
    """State for thinking tag tracking."""
    in_thinking: bool = False
    needs_prepend: bool = False
    reasoning_tokens: int = 0


# -----------------------------------------------------------------------------
# GPT-OSS Processing State
# -----------------------------------------------------------------------------

@cache
def _get_gpt_oss_encoding():
    """Get the GPT-OSS harmony encoding (cached)."""
    from openai_harmony import HarmonyEncodingName, load_harmony_encoding
    return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


@dataclass
class GptOssProcessResult:
    """Result of processing a GPT-OSS token."""
    text_to_emit: str | None = None  # Text to emit (None = skip)
    tool_call: ToolCallItem | None = None  # Completed tool call
    is_thinking_start: bool = False  # Emit <think> before text
    is_thinking_end: bool = False  # Emit </think> before text


class GptOssState:
    """
    Per-request state for GPT-OSS token processing.
    
    Wraps StreamableParser and handles thinking/tool call channel tracking.
    """
    
    def __init__(self):
        if not _check_openai_harmony():
            raise ImportError("openai_harmony required for GPT-OSS support")
        
        from openai_harmony import Role, StreamableParser
        encoding = _get_gpt_oss_encoding()
        self._stream = StreamableParser(encoding, role=Role.ASSISTANT)
        self._thinking = False
        self._current_tool_name: str | None = None
        self._tool_arg_parts: list[str] = []
    
    def process_token(self, token_id: int) -> GptOssProcessResult:
        """
        Process a GPT-OSS token and return the result.
        
        Args:
            token_id: The token ID to process
            
        Returns:
            GptOssProcessResult indicating what to emit
        """
        self._stream.process(token_id)
        
        delta = self._stream.last_content_delta
        ch = self._stream.current_channel
        recipient = self._stream.current_recipient
        
        result = GptOssProcessResult()
        
        # Handle tool call recipient changes
        if recipient != self._current_tool_name:
            if self._current_tool_name is not None:
                # Emit completed tool call
                tool_name = self._current_tool_name
                prefix = "functions."
                if tool_name.startswith(prefix):
                    tool_name = tool_name[len(prefix):]
                result.tool_call = ToolCallItem(
                    name=tool_name,
                    arguments="".join(self._tool_arg_parts).strip(),
                )
                self._tool_arg_parts = []
            self._current_tool_name = recipient
        
        # If inside a tool call, accumulate arguments
        if self._current_tool_name is not None:
            if delta:
                self._tool_arg_parts.append(delta)
            return result  # Don't emit text during tool call
        
        # Handle thinking channel
        if ch == "analysis" and not self._thinking:
            self._thinking = True
            result.is_thinking_start = True
        
        if ch != "analysis" and self._thinking:
            self._thinking = False
            result.is_thinking_end = True
        
        if delta:
            result.text_to_emit = delta
        
        return result
    
    def finish(self) -> GptOssProcessResult:
        """
        Called when generation finishes. Returns any pending state.
        """
        result = GptOssProcessResult()
        if self._thinking:
            result.is_thinking_end = True
            self._thinking = False
        return result

    @property
    def in_thinking(self) -> bool:
        """Whether currently in thinking mode."""
        return self._thinking


# -----------------------------------------------------------------------------
# TokenProcessor - Unified Token Processing Abstraction
# -----------------------------------------------------------------------------

@dataclass
class ProcessedToken:
    """Result of processing a single token."""
    text: str | None = None  # Text to emit (None = skip this token)
    tool_call: ToolCallItem | None = None  # Completed tool call to emit
    is_done: bool = False  # True if this is the final token


class TokenProcessor(ABC):
    """
    Abstract base for model-specific token processing.
    
    Constructed once per request, encapsulates all model-specific behavior
    so callers don't need to track model type or pass flags like is_gpt_oss.
    """
    
    @abstractmethod
    def process(self, token_id: int, token_text: str, is_first: bool) -> ProcessedToken:
        """
        Process a single token and return what should be emitted.
        
        Args:
            token_id: The raw token ID
            token_text: Decoded token text
            is_first: True if this is the first token of generation
            
        Returns:
            ProcessedToken indicating what to emit
        """
        ...
    
    @abstractmethod
    def finish(self) -> ProcessedToken | None:
        """
        Called when generation finishes. Returns any pending output.
        """
        ...
    
    @classmethod
    def for_model(
        cls,
        model: Model,
        model_id: str,
        needs_think_prepend: bool,
        tokenizer: TokenizerWrapper,
    ) -> TokenProcessor:
        """
        Factory method to create the appropriate TokenProcessor.
        
        This is the main entry point - callers pass model info once,
        then use the returned processor without worrying about model type.
        """
        from mlx_lm.models.gpt_oss import Model as GptOssModel
        
        is_gpt_oss = isinstance(model, GptOssModel)
        think_start = getattr(tokenizer, 'think_start', '<think>') if needs_think_prepend else None
        
        if is_gpt_oss and _check_openai_harmony():
            return GptOssTokenProcessor()
        elif "kimi" in model_id.lower():
            return KimiTokenProcessor(think_start)
        elif think_start:
            return ThinkingPrependProcessor(think_start)
        else:
            return PassthroughProcessor()


class PassthroughProcessor(TokenProcessor):
    """Default processor - passes tokens through unchanged."""
    
    def process(self, token_id: int, token_text: str, is_first: bool) -> ProcessedToken:
        return ProcessedToken(text=token_text)
    
    def finish(self) -> ProcessedToken | None:
        return None


class KimiTokenProcessor(TokenProcessor):
    """Processor for Kimi models - filters tool call section markers and handles thinking prepend."""
    
    def __init__(self, think_start: str | None = None):
        self._think_start = think_start
        self._prepended = False
    
    def process(self, token_id: int, token_text: str, is_first: bool) -> ProcessedToken:
        # Filter out tool call section markers
        if token_text in KIMI_FILTER_TOKENS:
            return ProcessedToken()  # Skip this token
        
        # Prepend thinking tag on first non-filtered token
        if self._think_start and not self._prepended:
            self._prepended = True
            return ProcessedToken(text=self._think_start + token_text)
        
        return ProcessedToken(text=token_text)
    
    def finish(self) -> ProcessedToken | None:
        return None


class ThinkingPrependProcessor(TokenProcessor):
    """Processor for thinking models - prepends think tag on first token."""
    
    def __init__(self, think_start: str):
        self._think_start = think_start
        self._prepended = False
    
    def process(self, token_id: int, token_text: str, is_first: bool) -> ProcessedToken:
        # Prepend thinking tag on first token
        if not self._prepended:
            self._prepended = True
            return ProcessedToken(text=self._think_start + token_text)
        return ProcessedToken(text=token_text)
    
    def finish(self) -> ProcessedToken | None:
        return None


class GptOssTokenProcessor(TokenProcessor):
    """Processor for GPT-OSS models - handles channel parsing and tool calls."""
    
    def __init__(self):
        self._state = GptOssState()
    
    def process(self, token_id: int, token_text: str, is_first: bool) -> ProcessedToken:
        result = self._state.process_token(token_id)
        
        # Handle tool call
        if result.tool_call:
            return ProcessedToken(tool_call=result.tool_call)
        
        # Build text output with thinking tags
        text_parts: list[str] = []
        if result.is_thinking_start:
            text_parts.append("<think>")
        if result.is_thinking_end:
            text_parts.append("</think>")
        if result.text_to_emit:
            text_parts.append(result.text_to_emit)
        
        if text_parts:
            return ProcessedToken(text="".join(text_parts))
        return ProcessedToken()  # Skip - no output yet
    
    def finish(self) -> ProcessedToken | None:
        result = self._state.finish()
        if result.is_thinking_end:
            return ProcessedToken(text="</think>")
        return None


# -----------------------------------------------------------------------------
# Stateless Helper Functions
# -----------------------------------------------------------------------------

def should_filter_kimi_token(token_text: str) -> bool:
    """Check if a token should be filtered out for Kimi models."""
    return token_text in KIMI_FILTER_TOKENS


def validate_single_tool(obj: dict[str, Any]) -> ToolCallItem:
    """Validate and convert a parsed tool call dict to ToolCallItem."""
    name = obj.get("name")
    args = obj.get("arguments")
    if name is not None and args is not None and isinstance(name, str):
        return ToolCallItem(name=name, arguments=json.dumps(args))
    raise ValidationError


def parse_tool_call_result(
    tool_text: str,
    tool_parser: Callable[[str], dict[str, Any] | list[dict[str, Any]]],
) -> list[ToolCallItem] | None:
    """
    Parse accumulated tool call text into ToolCallItems.
    
    Returns a list of ToolCallItems on success, or None if parsing failed.
    The caller should handle the failure case (e.g., emit raw text).
    """
    try:
        parsed = tool_parser(tool_text.strip())
        if isinstance(parsed, list):
            return [validate_single_tool(tool) for tool in parsed]
        return [validate_single_tool(parsed)]
    except (json.JSONDecodeError, ValidationError, ValueError, AttributeError):
        return None


@dataclass
class ToolCallProcessResult:
    """Result of processing a token for tool calls."""
    consumed: bool  # True if token was consumed by tool call processing
    completed_tools: list[ToolCallItem] | None = None  # Tools if parsing succeeded
    failed_text: str | None = None  # Raw text if parsing failed
    interrupted_text: str | None = None  # Partial text if finish_reason is set


def process_tool_call_token(
    token_text: str,
    state: ToolCallState,
    tool_call_start: str,
    tool_call_end: str,
    tool_parser: Callable[[str], dict[str, Any] | list[dict[str, Any]]],
    finish_reason: str | None = None,
) -> ToolCallProcessResult:
    """
    Process a token for tool call handling.
    
    This is a stateful function - it modifies `state` to track tool call accumulation.
    
    Args:
        token_text: The decoded token text
        state: Tool call accumulation state
        tool_call_start: Token marking start of tool call
        tool_call_end: Token marking end of tool call  
        tool_parser: Function to parse tool call text
        finish_reason: If set, indicates generation is finishing
        
    Returns:
        ToolCallProcessResult indicating how the token was handled
    """
    # Check for tool call start
    if token_text == tool_call_start:
        state.in_tool_call = True
        state.tool_call_parts = []
        return ToolCallProcessResult(consumed=True)
    
    # Check for tool call end
    if state.in_tool_call and token_text == tool_call_end:
        state.in_tool_call = False
        tool_text = "".join(state.tool_call_parts)
        tools = parse_tool_call_result(tool_text, tool_parser)
        state.tool_call_parts = []
        
        if tools is not None:
            return ToolCallProcessResult(consumed=True, completed_tools=tools)
        else:
            # Parsing failed - return raw text for emission
            return ToolCallProcessResult(
                consumed=True,
                failed_text=tool_call_start + tool_text + tool_call_end
            )
    
    # Inside tool call - accumulate
    if state.in_tool_call:
        state.tool_call_parts.append(token_text)
        
        # Check if we're being interrupted by finish
        if finish_reason is not None:
            interrupted_text = tool_call_start + "".join(state.tool_call_parts)
            state.in_tool_call = False
            state.tool_call_parts = []
            return ToolCallProcessResult(consumed=True, interrupted_text=interrupted_text)
        
        return ToolCallProcessResult(consumed=True)
    
    return ToolCallProcessResult(consumed=False)


def update_thinking_state(
    token_text: str,
    state: ThinkingState,
    think_start: str | None,
    think_end: str | None,
) -> None:
    """
    Update thinking state based on token.
    
    Modifies state.in_thinking and increments state.reasoning_tokens.
    """
    if think_start is not None and token_text == think_start:
        state.in_thinking = True
    elif think_end is not None and token_text == think_end:
        state.in_thinking = False
    
    if state.in_thinking:
        state.reasoning_tokens += 1


def check_stop_sequence(
    accumulated_text: str,
    token_text: str,
    stop_sequences: list[str],
) -> tuple[str, bool]:
    """
    Check if a stop sequence has been reached.
    
    Args:
        accumulated_text: All text generated so far (including current token)
        token_text: The current token text
        stop_sequences: List of stop sequences to check
        
    Returns:
        Tuple of (text_to_emit, should_stop)
        - text_to_emit: The text that should be emitted (truncated if stop found)
        - should_stop: True if a stop sequence was found
    """
    for stop_seq in stop_sequences:
        if stop_seq in accumulated_text:
            stop_index = accumulated_text.find(stop_seq)
            text_before_stop = accumulated_text[:stop_index]
            # Calculate what portion of the current token should be emitted
            chunk_start = len(accumulated_text) - len(token_text)
            text_to_emit = text_before_stop[chunk_start:]
            return (text_to_emit, True)
    
    return (token_text, False)
