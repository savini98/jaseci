"""MTRuntime (Meaning Typed Intermediate Representation) module for JacLang runtime library."""
import inspect;
import json;
import from jaclang.jac0core.mtp { Info, MethodInfo, FunctionInfo }
import from types { MethodType }
import from typing { Callable, get_type_hints }
import from byllm.schema { json_to_instance, type_to_schema }
import from byllm.exceptions { ConfigurationError, OutputConversionError }
import from byllm.config_loader { get_byllm_config }
import from byllm.types {
    LiteLLMMessage,
    Media,
    Message,
    MessageRole,
    MessageType,
    Text,
    Tool,
    ToolCallResultMsg
}
import from byllm.mcp { McpTool }
import from byllm.memory { Memory, MemoryGraph, MemoryPolicy }

"""Intermediate representation for LLM function calls.

Captures the callable, its arguments, and call parameters to construct
an MTRuntime instance on demand via the runtime property.
"""
obj MTIR {
    has caller: Callable,
        args: dict[(int | str, object)],
        call_params: dict[(str, object)],
        ir_info: Info;

    @property
    def runtime -> MTRuntime {
        return MTRuntime.factory(self);
    }
}

"""Runtime context for LLM interactions.

Manages the message history, tools, response type, and streaming state
for a single LLM invocation. Created via the factory method from an MTIR.
"""
obj MTRuntime {
    has messages: list[MessageType],
        resp_type: type | None,
        stream: bool,
        tools: list[Tool | McpTool],
        call_params: dict[(str, object)],
        mtir: MTIR | None = None,
        # IDs of byllm-injected messages excluded from the persisted view.
        _scaffolding_ids: list[int] = [],
        # Caller's list bound via `by llm(conversation=<list>)`, written
        # back in place after the ReAct loop. None when not conversational.
        _conversation_list: object = None,
        # Set by `dispatch_streaming` when streaming a structured (typed)
        # response. Holds the parsed `resp_type` instance once the stream
        # completes; surfaced to callers via `StructuredStream.parsed`.
        streaming_result: object = None,
        # Per-invocation memory graph. Populated by the factory when the
        # caller passes `memory=Memory(...)` to `by llm()`. Tool results
        # and messages are auto-recorded into this graph as the agent
        # runs; the caller inspects it via `Memory.graph` after the call.
        # Stays None for callers who don't opt in.
        memory: MemoryGraph | None = None,
        # Strategy interface (Stage 3+) for pruning, compaction, and
        # materialization. Set from `Memory.policy` in the factory.
        # When non-None, byLLM dispatches to it at iteration boundaries
        # instead of running its own inline compaction logic.
        memory_policy: MemoryPolicy | None = None;

    static def factory(mtir: MTIR) -> MTRuntime;
    def dispatch_params -> dict[str, object];
    def add_message(message: MessageType) -> None;
    def add_scaffolding_message(message: MessageType) -> None;
    def get_msg_list -> list[dict[(str, object)] | LiteLLMMessage];
    def get_history_for_persist -> list[MessageType];
    def write_back_conversation -> None;
    def parse_response(response: str) -> object;
    def get_tool(tool_name: str) -> Tool | McpTool | None;
    def get_tool_list -> list[dict];
    def get_output_schema -> dict | None;
}


