glob SYSTEM_PERSONA = """\
    This is a task you must complete by returning only the output.
    Do not include explanations, code, or extra text—only the result.
    """,
     INSTRUCTION_TOOL = """
    Use the tools provided to reach the goal. Call one tool at a time with \
    proper args—no explanations, no narration. Think step by step, invoking tools \
    as needed. When done, always call finish_tool(output) to return the final
    output. Only use tools.
    """;


"""Normalise an assistant message's tool_calls into the
[{id, name, args}] shape ContextMessage.tool_calls stores.

Handles three observed input shapes:
  - LiteLLM-style attribute access (msg.tool_calls -> obj.id /
    obj.function.name / obj.function.arguments as JSON str)
  - Dict-style fallback (tool_calls is a list of {id, function: {...}})
  - Missing / None (returns [])

Used by both MTRuntime.factory's bulk-record loop and add_message so
the graph captures the tool_use_id ↔ tool_name link that
materialize() relies on to rebuild a valid LLM message list when
pruning has acted on individual tool results.
"""
def _extract_tool_calls(msg: object) -> list {
    raw = getattr(msg, "tool_calls", None);
    if raw is None and isinstance(msg, dict) {
        raw = msg.get("tool_calls");
    }
    if not raw {
        return [];
    }
    out: list = [];
    for tc in raw {
        tc_id: str = "";
        tc_name: str = "";
        tc_args: dict = {};
        # Attribute style first (LiteLLM / OpenAI client objects).
        tc_id = str(getattr(tc, "id", "") or "");
        fn = getattr(tc, "function", None);
        if fn is not None {
            tc_name = str(getattr(fn, "name", "") or "");
            raw_args = getattr(fn, "arguments", None);
            if isinstance(raw_args, str) and raw_args {
                try {
                    tc_args = json.loads(raw_args);
                } except Exception {
                    tc_args = {"_raw": raw_args};
                }
            } elif isinstance(raw_args, dict) {
                tc_args = raw_args;
            }
        }
        # Dict-shape fallback if attribute access didn't fill anything.
        if not tc_id and isinstance(tc, dict) {
            tc_id = str(tc.get("id", "") or "");
            fn_dict = tc.get("function") or {};
            if isinstance(fn_dict, dict) {
                tc_name = str(fn_dict.get("name", "") or "");
                raw_args2 = fn_dict.get("arguments");
                if isinstance(raw_args2, str) and raw_args2 {
                    try {
                        tc_args = json.loads(raw_args2);
                    } except Exception {
                        tc_args = {"_raw": raw_args2};
                    }
                } elif isinstance(raw_args2, dict) {
                    tc_args = raw_args2;
                }
            }
        }
        if tc_id {
            out.append({"id": tc_id, "name": tc_name, "args": tc_args});
        }
    }
    return out;
}

"""Create an MTRuntime instance from a callable and its arguments.

Builds the message history with system persona, user input, and any media.
Prepares tools including the finish_tool for ReAct-style interactions.
"""
impl MTRuntime.factory(mtir: MTIR) -> MTRuntime {
    # Prepare the tools for the LLM call.
    caller = mtir.caller;
    args = mtir.args;
    call_params = mtir.call_params;
    ir_info = mtir.ir_info;
    # Warn and remove deprecated 'method' parameter.
    if "method" in call_params {
        import warnings;
        warnings.warn(
            "'method' parameter is deprecated and has no effect. "
            "For tool usage, just pass tools=[...] directly to by llm().",
            DeprecationWarning,
            stacklevel=2
        );
        del call_params["method"];
    }
    # Check if we have MTIR info (Jac mode) or need runtime introspection (Python lib mode)
    has_mtir = ir_info is not None;
    # Prepare tools with or without MTIR info
    raw_tools = call_params.get("tools", []);
    tools = [];
    local_idx = 0;
    for func in raw_tools {
        # Already tool-like (e.g., McpTool) — pass through unchanged.
        if func?.get_json_schema and func?.get_name and func?.is_finish_tool {
            tools.append(func);
            continue;
        }
        if has_mtir {
            info = ir_info.tools[local_idx] if local_idx < len(ir_info.tools) else None;
            tools.append(Tool(func, info=info));  # type: ignore
        } else {
            tools.append(Tool(func));  # type: ignore
        }
        local_idx += 1;
    }
    # Construct the input information from the arguments.
    inputs_detail: list[str] = [];
    media_inputs: list = [];  # Can be list[Media] or list[tuple[Media, str]] depending on mode
    if has_mtir {
        # MTIR mode: use compile-time extracted param info with semantic strings
        params = [param for param in ir_info.params];
        for (key, value) in args.items() {
            idx = list(args.keys()).index(key);
            if isinstance(value, Media) {
                media_inputs.append((value, params[idx].semstr or ""));
                continue;
            }

            if isinstance(key, str) {
                inputs_detail.append(f"{key} = {(value, params[idx].semstr or "")}");
            } else {
                # TODO: Handle *args, **kwargs properly.
                if key < len(params) {
                    inputs_detail.append(
                        f"{params[key].name} = {(value, params[idx].semstr or "")}"
                    );
                } else {
                    inputs_detail.append(f"arg = {(value, params[idx].semstr or "")}");
                }
            }
        }
    } else {
        # Python library mode: use runtime introspection (old behavior)
        param_names = list(inspect.signature(caller).parameters.keys());
        for (key, value) in args.items() {
            if isinstance(value, Media) {
                # Use tuple format for consistency, but without semantic strings
                media_inputs.append((value, ""));
                continue;
            }

            if isinstance(key, str) {
                inputs_detail.append(f"{key} = {value}");
            } else {
                # TODO: Handle *args, **kwargs properly.
                if key < len(param_names) {
                    inputs_detail.append(f"{param_names[key]} = {value}");
                } else {
                    inputs_detail.append(f"arg = {value}");
                }
            }
        }
    }
    incl_info = call_params.get("incl_info");
    if incl_info and isinstance(incl_info, dict) {
        for (key, value) in incl_info.items() {
            if isinstance(value, Media) {
                # Always use tuple format for consistency
                media_inputs.append((value, ""));
            } else {
                if has_mtir {
                    inputs_detail.append(f"{key} = {(value, "")}");
                } else {
                    inputs_detail.append(f"{key} = {value}");
                }
            }
        }
    }
    if isinstance(caller, MethodType) {
        if has_mtir {
            inputs_detail.insert(
                0,
                f"self = {(
                    caller.__self__,
                    ir_info.parent_class.semstr if ir_info.parent_class else ''
                )}"
            );
        } else {
            inputs_detail.insert(0, f"self = {caller.__self__}");
        }
    }
    # System prompt: Config from jac.toml or default SYSTEM_PERSONA
    system_content = SYSTEM_PERSONA;
    try {
        import from jaclang.project.config { find_project_root }
        import from pathlib { Path }

        # Get the file path of the caller's module
        caller_file = inspect.getfile(caller);
        project_root_result = find_project_root(Path(caller_file).parent);
        project_dir = project_root_result[0] if project_root_result else None;

        config_system_prompt = get_byllm_config(project_dir).get_system_prompt();
        if config_system_prompt {
            system_content = config_system_prompt;
        }
    } except Exception { }

    # Conversational mode: `by llm(conversation=<list>)` splices the
    # caller's prior conversation between SYSTEM and USER. Entries on input
    # may be Messages or dicts; write-back always emits dicts.
    conv_arg = call_params.get("conversation");
    conv_msgs: list[MessageType] = [];
    conv_list_ref: object = None;
    if isinstance(conv_arg, list) {
        conv_list_ref = conv_arg;
        for item in conv_arg {
            m: Any = item;
            if isinstance(m, Message) {
                conv_msgs.append(m);
            } elif isinstance(m, dict) {
                reconstructed: Any = Message.from_dict(m);
                conv_msgs.append(reconstructed);
            } else {
                conv_msgs.append(m);
            }
        }
    }

    # Prepare the messages for the LLM call.
    sys_msg = Message(
        role=MessageRole.SYSTEM,
        content=system_content + (INSTRUCTION_TOOL if tools else ""),
    );
    user_msg = Message(
        role=MessageRole.USER,
        content=[
            Text(Tool.get_func_description(caller) + "\n\n" + "\n".join(inputs_detail)),
            *media_inputs,

        ],
    );
    messages: list[MessageType] = [sys_msg, *conv_msgs, user_msg];

    # Prepare return type.
    return_type = get_type_hints(caller).get("return");
    is_streaming = bool(call_params.get("stream", False));
    # Typed+streaming flow:
    #
    # - Without tools: dispatch_streaming sends `response_format` straight
    #   to the provider, the model emits structured JSON, and
    #   StreamingFieldExtractor surfaces tokens from the designated text
    #   field while the rest of the JSON is buffered for full parsing on
    #   completion.
    # - With tools: the ReAct loop runs as usual; finish_tool is still
    #   injected so the loop has a clean termination signal. When
    #   finish_tool fires, `_stream_final_answer` detects typed mode and
    #   makes a streaming follow-up call with tools cleared (so the model
    #   produces structured JSON via response_format) — that follow-up is
    #   what gets streamed token-by-token.
    if len(tools) > 0 {
        finish_tool = Tool.make_finish_tool(return_type or str);
        tools.append(finish_tool);
    }
    mt_run = MTRuntime(
        messages=messages,
        tools=tools,
        resp_type=return_type,
        stream=is_streaming,
        call_params=call_params,
        mtir=mtir
    );

    # In conversational mode, the auto-SYSTEM is regenerated from sem each
    # turn — exclude it from persistence and bind the caller's list for
    # write-back.
    if conv_list_ref is not None {
        mt_run._scaffolding_ids.append(id(sys_msg));
        mt_run._conversation_list = conv_list_ref;
    }

    # Context: when the caller passed `context=Context(...)` to `by llm()`,
    # ask the Context holder for its (possibly persisted) ContextGraph and
    # attach it to mt_run so downstream auto-recording hooks can find it.
    context_obj = call_params.get("context");
    if isinstance(context_obj, Context) {
        graph: ContextGraph = context_obj.get_or_create_graph();
        mt_run.context = graph;
        # Stage 3: propagate the policy from the Context holder onto
        # mt_run so the loop can dispatch to it without keeping a
        # reference to the holder itself.
        mt_run.context_policy = context_obj.policy;
        # Bump the per-call turn counter on the graph so generated IDs
        # carry the right turn even when the agent itself hasn't tracked
        # it (the agent code can override later via its own turn logic).
        graph.turn_count += 1;
        current_turn: int = graph.turn_count;
        # Auto-record the user-side messages already on mt_run.messages
        # (system + conversation history + new user msg). Keeps the graph
        # in sync from turn 0 without touching add_message hooks.
        for m in mt_run.messages {
            role_str = "";
            content_str = "";
            role_attr = getattr(m, "role", None);
            if role_attr is not None {
                role_str = str(getattr(role_attr, "value", role_attr));
            }
            raw_content = getattr(m, "content", "");
            if isinstance(raw_content, str) {
                content_str = raw_content;
            } else {
                content_str = str(raw_content);
            }
            if role_str {
                graph.record_message(
                    role=role_str,
                    turn=current_turn,
                    content=content_str,
                    tool_calls=_extract_tool_calls(m)
                );
            }
        }
    }

    return mt_run;
}

"""Build the parameter dict for the LLM API call."""
impl MTRuntime.dispatch_params -> dict[str, object] {
    params = {
        "messages": self.get_msg_list(),
        "tools": self.get_tool_list() or None,
        "response_format": self.get_output_schema(),
        "temperature": self.call_params.get("temperature", 0.7),
        # "max_tokens": self.call_params.get("max_tokens", 100),
        # "top_k": self.call_params.get("top_k", 50),
        # "top_p": self.call_params.get("top_p", 0.9),
    };
    return params;
}

"""Add a message to the conversation history.

When the caller opted into the memory layer via `context=Context(...)`,
also record the message in the ContextGraph as a ContextMessage. We skip
ToolCallResultMsg here because those are recorded as ContextToolResult
(richer schema with args, full_content, tr_id) at the dispatch site.
"""
impl MTRuntime.add_message(message: MessageType) -> None {
    self.messages.append(message);
    graph = self.context;
    if graph is None {
        return;
    }
    if isinstance(message, ToolCallResultMsg) {
        return;
    }
    role_str = "";
    content_str = "";
    role_attr = getattr(message, "role", None);
    if role_attr is not None {
        role_str = str(getattr(role_attr, "value", role_attr));
    }
    raw_content = getattr(message, "content", "");
    if isinstance(raw_content, str) {
        content_str = raw_content;
    } else {
        content_str = str(raw_content);
    }
    if role_str {
        graph.record_message(
            role=role_str,
            turn=graph.turn_count,
            content=content_str,
            tool_calls=_extract_tool_calls(message)
        );
    }
}


impl MTRuntime.add_scaffolding_message(message: MessageType) -> None {
    self.messages.append(message);
    self._scaffolding_ids.append(id(message));
}

"""Return the messages in a format suitable for LLM API."""
impl MTRuntime.get_msg_list -> list[dict[str, object] | LiteLLMMessage] {
    # Context-policy bridge: when the caller wired
    # `context=Context(policy=...)` and the policy's `materialize()`
    # returns a non-None list, use that as the source instead of
    # self.messages. The substitution only affects what gets
    # serialized to the LLM this call — self.messages itself is
    # never mutated, so the canonical conversation history (and the
    # caller's `conversation=` list) stays intact.
    #
    # Wrapped in try/except: materialize is opt-in extra logic, never
    # let a bug there tank the LLM call. Fall back to the canonical
    # history on any error.
    source = self.messages;
    if self.context_policy is not None and self.context is not None {
        try {
            mat = self.context_policy.materialize(self.context, self);
            if mat is not None {
                source = mat;
            }
        } except Exception {
            # Materialize is opt-in. If a custom policy raises here we
            # fall back to the canonical history rather than blocking
            # the LLM call.
        }
    }
    return [
        msg.to_dict() if isinstance(msg, Message) else msg for msg in source
    ];
}


impl MTRuntime.write_back_conversation -> None {
    if self._conversation_list is None {
        return;
    }
    out: list = [];
    for msg in self.get_history_for_persist() {
        m: Any = msg;
        if hasattr(m, "to_dict") {
            out.append(m.to_dict());
        } elif isinstance(m, dict) {
            out.append(dict(m));
        } elif hasattr(m, "model_dump") {
            try {
                out.append(m.model_dump());
            } except Exception {
                out.append({"raw": str(m)});
            }
        }
    }
    self._conversation_list[:] = out;
}


# Test against finish_tool both at object level and dict level.
def _is_finish_tool_call(tc: Any) -> bool {
    if tc?.function and hasattr(tc.function, "name") {
        return tc.function.name == "finish_tool";
    }
    if isinstance(tc, dict) {
        return (tc.get("function") or {}).get("name", "") == "finish_tool";
    }
    return False;
}


impl MTRuntime.get_history_for_persist -> list[MessageType] {
    scaff_ids = set(self._scaffolding_ids);
    out: list = [];
    for msg in self.messages {
        if id(msg) in scaff_ids {
            continue;
        }
        # Skip non-Message artifacts (e.g. MockLLM raw strings).
        if not (
            isinstance(msg, Message)
            or isinstance(msg, dict)
            or hasattr(msg, "to_dict")
            or hasattr(msg, "model_dump")
        ) {
            continue;
        }
        if isinstance(msg, ToolCallResultMsg) and msg.name == "finish_tool" {
            continue;
        }
        tool_calls = msg?.tool_calls;
        if tool_calls is None and isinstance(msg, dict) {
            tool_calls = msg.get("tool_calls");
        }
        if tool_calls and all(_is_finish_tool_call(tc) for tc in tool_calls) {
            continue;
        }
        out.append(msg);
    }
    return out;
}

"""Parse the LLM response string into the expected return type."""
impl MTRuntime.parse_response(response: str) -> object {
    # To use validate_json the string should contains quotes.
    #     example: '"The weather at New York is sunny."'
    # but the response from LLM will not have quotes, so
    # we need to check if it's string and return early.
    if self.resp_type is None or self.resp_type is str or response.strip() == "" {
        return response;
    }
    if self.resp_type {
        try {
            json_dict = json.loads(response);
            return json_to_instance(json_dict, self.resp_type);
        } except OutputConversionError {
            raise;
        } except Exception as e {
            err = OutputConversionError(
                f"Failed to convert LLM output to '{self.resp_type.__name__}': {e}"
            );
            err.raw_output = response;
            raise err;
        }
    }
    return response;
}

"""Get a tool by name. Works for both Tool and McpTool."""
impl MTRuntime.get_tool(tool_name: str) -> Tool | McpTool | None {
    for tool in self.tools {
        if tool.get_name() == tool_name {
            return tool;
        }
    }
    return None;
}

"""Return the tools as JSON schemas for the LLM API."""
impl MTRuntime.get_tool_list -> list[dict] {
    return [tool.get_json_schema() for tool in self.tools];
}

"""Return the JSON schema for the response type, if applicable."""
impl MTRuntime.get_output_schema -> dict | None {
    assert (
        len(self.tools) == 0 or self.get_tool("finish_tool") is not None
    );  #Finish tool should be present in the tools list.
    if len(self.tools) == 0 and self.resp_type {
        if self.resp_type is str {
            return None;  # Strings are default and not using a schema.
        }
        # Use MTIR info if available, otherwise None (Python library mode)
        return_type_info = self.mtir.ir_info.return_type
            if (self.mtir and self.mtir.ir_info)
            else None;
        return type_to_schema(resp_type=self.resp_type, info=return_type_info);
    }
    # If the are tools, the final output will be sent to the finish_tool
    # thus there is no output schema.
    return None;
}
