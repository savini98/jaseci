"""Post-initialization hook for MockLLM.

Configures logging to stdout.
"""
impl MockLLM.postinit -> None {
    # Route mock output to stdout via loguru's sink API. Skipped under the stdlib
    # fallback logger (no `add`/`remove`), so MockLLM works without the `llm` capability.
    import from jaclang.byllm._optdeps.loguru { HAS_LOGURU }
    if HAS_LOGURU {
        logger.remove();  # remove default stderr sink
        logger.add(sys.stdout);
    }
}

"""Dispatch the mock LLM call without streaming.

Pops the next output from the configured 'outputs' list. Each entry may be a
plain MockToolCall/string or a (MockToolCall, usage_dict) tuple to inject
token-usage metadata for compaction threshold tests. The tuple form is only
recognised when the first element is a MockToolCall and the second is a dict
with only known usage keys - this avoids colliding with real tuple return types.
"""
impl MockLLM.dispatch_no_streaming(mt_run: MTRuntime) -> CompletionResult {
    params = self.make_model_params(mt_run);
    # Record the prompt this call saw (joined message contents) before adding
    # the mock's own assistant turn, so tests can inspect retry feedback.
    self.seen_prompts.append(
        "\n".join([str(getattr(m, "content", m)) for m in mt_run.messages])
    );
    output = self.config["outputs"].pop(0);  # type: ignore
    # Error response: raise before recording an assistant turn, mirroring a real
    # pre-response dispatch failure (no assistant message is added). Used to
    # verify that non-OutputConversionError failures propagate without a retry.
    if isinstance(output, MockError) {
        raise output.error;
    }
    message = "{'role': 'assistant', 'content': {'text': 'Mock response'}}";
    mt_run.add_message(message);
    # Unpack (payload, usage) tuple when tests need to inject usage metadata.
    # Both conditions must hold to avoid colliding with real tuple return types:
    #   1. first element is a MockToolCall or list[MockToolCall] (never a real LLM output type)
    #   2. second element is a dict with only known usage keys (or empty)
    _usage_keys = ["prompt_tokens", "completion_tokens", "total_tokens"];
    usage: dict[str, object] = {};
    if isinstance(output, tuple) and len(output) == 2 {
        first = output[0];
        second = output[1];
        first_is_tool = (
            isinstance(first, MockToolCall)
            or (
                isinstance(first, list)
                and len(first) > 0
                and isinstance(first[0], MockToolCall)
            )
        );
        second_is_usage = isinstance(second, dict)
        and all(k in _usage_keys for k in second.keys());
        if first_is_tool and second_is_usage {
            output = first;
            usage = second;
        }
    }
    # Raw text response: route through parse_response exactly like the real
    # dispatch, so structured types convert and malformed JSON raises
    # OutputConversionError (which drives the typed-output retry loop).
    if isinstance(output, MockRawResponse) {
        return CompletionResult(
            output=mt_run.parse_response(output.content), tool_calls=[], usage=usage
        );
    }
    # Support batch tool calls: list[MockToolCall] -> multiple ToolCalls in one response.
    if isinstance(output, list)
    and len(output) > 0
    and isinstance(output[0], MockToolCall) {
        tool_calls = [o.to_tool_call() for o in output];
        self.log_info(
            f"Mock LLM call completed with {len(tool_calls)} batch tool calls"
        );
        return CompletionResult(output=None, tool_calls=tool_calls, usage=usage);
    }
    if isinstance(output, MockToolCall) {
        self.log_info(
            f"Mock LLM call completed with tool call:\n{output.to_tool_call()}"
        );
        return CompletionResult(
            output=None, tool_calls=[output.to_tool_call()], usage=usage
        );
    }
    log_params = self.format_prompt(params);
    params.pop("api_key", None);
    if bool(self.config.get("show_params", False)) {
        self.log_info(f"Mock LLM Call Parameters: {params}");
    } else {
        self.log_info(
            f"Mock LLM call completed with response: {output} with params: {log_params}"
        );
    }
    return CompletionResult(output=output, tool_calls=[], usage=usage);
}

"""Async dispatch for the mock LLM: reuse the sync mock (no real I/O).

Lets `async def ... by llm()` exercise the mock outputs and the async typed-output
retry path without hitting a real provider.
"""
impl MockLLM.adispatch_no_streaming(mt_run: MTRuntime) -> CompletionResult {
    return self.dispatch_no_streaming(mt_run);
}

"""Dispatch the mock LLM call with streaming.

Simulates streaming by yielding random-sized chunks with artificial delays.
"""
impl MockLLM.dispatch_streaming(mt_run: MTRuntime) -> Generator[str, None, None] {
    output = self.config["outputs"].pop(0);  # type: ignore

    # Mirror BaseLLM.dispatch_streaming's structured-streaming behavior so
    # MockLLM-backed tests exercise the typed+stream path deterministically.
    # Handler-driven calls stream raw tokens (see BaseLLM.dispatch_streaming).
    is_structured = (
        mt_run.resp_type is not None
        and mt_run.resp_type is not str
        and mt_run.call_params.get("stream_handler") is None
    );
    explicit_field = (
        mt_run.call_params.get("streaming_text_field")
        or self.call_params.get("streaming_text_field")
    );
    text_field = resolve_streaming_text_field(
        explicit_field,
        mt_run.get_output_schema() if is_structured else None
    );
    extractor = (
        StreamingFieldExtractor(field_name=text_field)
            if is_structured
            else None
    );

    if mt_run.stream {
        while output {
            chunk_len = random.randint(3, 10);
            chunk = output[:chunk_len];
            if is_structured and extractor is not None {
                tokens = extractor.feed(chunk);
                if tokens {
                    yield tokens;
                }
            } else {
                yield chunk;
            }
            time.sleep(random.uniform(0.01, 0.05));  # Simulate network delay
            output = output[chunk_len:];
        }
    }

    # Post-stream: parse the buffered structured JSON and store the typed
    # instance on mt_run for the caller to retrieve.
    if is_structured and extractor is not None {
        raw_text: str = extractor.get_raw().strip();
        target_type = mt_run.resp_type;
        if raw_text and target_type is not None {
            try {
                parsed_dict = json.loads(raw_text);
                mt_run.streaming_result = json_to_instance(
                    parsed_dict, target_type
                );
            } except Exception as e {
                logger.warning(
                    "MockLLM structured streaming: failed to parse final JSON "
                    f"({type(e).__name__}: {e}). raw={raw_text[:200]!r}"
                );
                mt_run.streaming_result = None;
            }
        }
    }

    # Record a usage entry so the end-of-stream usage event fires.
    self._usage_history.append({});
}
