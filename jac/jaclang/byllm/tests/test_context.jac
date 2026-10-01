"""Context management wiring: `by llm(context=Context(...))`.

byLLM records every message and tool result of a run into a ContextGraph and asks
the attached ContextPolicy, before each model call, for the view the model sees.
These tests cover the recording and the overlay; the scoring policy has its own file.
"""

import from jaclang.byllm.lib { MockLLM }
import from jaclang.byllm.context {
    COMPACTION_TAG,
    Context,
    ContextCompactionSummary,
    ContextGraph,
    ContextPolicy,
    EvictionDecision,
    FOLDED_RESULT_NOTE,
    SOFT_TRIM_MARKER,
    apply_eviction_decision,
    estimate_tokens,
    managed_tool_results,
    materialize_overlay,
    message_role,
    message_text,
    tool_call_id_of
}
import from jaclang.byllm.types { Message, MessageRole, Text, ToolCallResultMsg }
import from support {
    call,
    finish,
    mk_run,
    say,
    sent_messages,
    stream_events,
    events_of
}


glob ROWS: str = "\n".join(
         [f"row_{str(i).zfill(4)} value={i * 7}" for i in range(400)]
     );


"""A tool whose result is long enough that a head/tail trim visibly shortens it."""
def lookup(key: str) -> str {
    return f"{key}:\n{ROWS}";
}


"""The ids of the tool calls an assistant message carries, whatever their shape."""
def tool_call_ids(message: dict[str, any]) -> list[str] {
    out: list[str] = [];
    for tc in (message.get("tool_calls") or []) {
        if isinstance(tc, dict) {
            out.append(str(tc.get("id", "")));
        } else {
            out.append(str(getattr(tc, "id", "")));
        }
    }
    return out;
}


"""The tool message for `call_id` in one sent conversation."""
def tool_message(messages: list[dict[str, any]], call_id: str) -> dict[str, any] {
    found = [
        m
        for m in messages
        if m.get("role") == "tool" and m.get("tool_call_id") == call_id
    ];
    assert len(found) == 1 , f"expected one tool message for {call_id}, got {found}";
    return found[0];
}


"""A policy that head/tail-trims every full result longer than 2 * keep chars on
each pass after the first, through apply_eviction_decision, and overlays the view."""
obj TrimPolicy(ContextPolicy) {
    has keep: int = 120,
        passes: int = 0;

    override def should_prune(cg: ContextGraph, mt_run: object, model: object) -> bool {
        return True;
    }

    override def prune(cg: ContextGraph, mt_run: object, model: object) -> int {
        self.passes += 1;
        changed = 0;
        for tr in cg.tool_results() {
            if tr.truncation_state == "full" and len(tr.full_content) > 2 * self.keep {
                freed = apply_eviction_decision(
                    EvictionDecision(
                        tool_result=tr,
                        kind="soft_trim",
                        head_chars=self.keep,
                        tail_chars=self.keep
                    )
                );
                if freed > 0 {
                    changed += 1;
                }
            }
        }
        return changed;
    }

    override def materialize(cg: ContextGraph, mt_run: object) -> list[any] | None {
        return materialize_overlay(cg, mt_run, []);
    }
}


"""A policy that trims nothing and only overlays the view."""
obj ViewPolicy(ContextPolicy) {
    override def materialize(cg: ContextGraph, mt_run: object) -> list[any] | None {
        return materialize_overlay(cg, mt_run, []);
    }
}


"""Run the two-lookup script through `agent` with or without a Context."""
def run_agent(llm: MockLLM, ctx: Context | None) -> str {
    def agent(task: str) -> str by llm(tools=[lookup], context=ctx);
    return agent("find alpha and beta");
}


def two_lookups -> list[any] {
    return [
        call("lookup", {"key": "alpha"}, call_id="call_a"),
        call("lookup", {"key": "beta"}, call_id="call_b"),
        finish("done")
    ];
}


test "context=Context() records tool results and messages into the graph" {
    llm = MockLLM(outputs=two_lookups());
    ctx = Context();

    assert run_agent(llm, ctx) == "done";

    cg = ctx.graph;
    assert cg is not None , "the Context exposes the graph after the call";
    results = cg.tool_results();
    assert [tr.call_id for tr in results] == ["call_a", "call_b"] , (
        f"one ContextToolResult per executed tool, keyed by the mock's call ids: {results}"
    );
    assert [tr.tool_name for tr in results] == ["lookup", "lookup"];
    assert results[0].full_content == lookup("alpha");
    assert results[0].visible_content == results[0].full_content , "recorded at full fidelity";
    assert results[0].truncation_state == "full" and results[0].visible;
    assert results[0].tr_id.startswith("tr_") and "lookup" in results[0].tr_id;

    messages = cg.messages();
    message_roles = [m.role for m in messages];
    assert message_roles[:2] == ["system", "user"] , f"initial messages first: {message_roles}";
    assert message_roles.count("assistant") == 3 , (
        f"one assistant message per model call (two tool calls and the finish): {message_roles}"
    );
    # Every transcript item is a turn, tool results included (the AppWorld
    # adapter numbers them the same way), so the counter covers both kinds.
    assert cg.turn_count == len(messages) + len(results) and cg.turn_count > 0;
    turns = [m.turn for m in messages];
    assert turns == sorted(turns) and len(set(turns)) == len(turns) , f"message turns are distinct and increasing: {turns}";

    assistant_turns = [
        m.turn
        for m in messages
        if m.role == "assistant"
    ];
    assert results[0].produced_at_turn == assistant_turns[0] + 1 , (
        "a tool result takes the turn right after the assistant message that issued it"
    );
    assert results[1].produced_at_turn == assistant_turns[1] + 1;
    first_call = [
        m
        for m in messages
        if m.role == "assistant"
    ][0];
    assert [tc["id"] for tc in first_call.tool_calls] == ["call_a"] , (
        f"the assistant message keeps its tool_calls as {{id, name, args}}: {first_call.tool_calls}"
    );
    assert first_call.tool_calls[0]["name"] == "lookup";
    assert first_call.tool_calls[0]["args"] == {"key": "alpha"};
}


test "a plain answer is recorded once as the assistant message" {
    llm = MockLLM(outputs=[say("hello there")]);
    ctx = Context(policy=ContextPolicy());

    def greet(name: str) -> str by llm(context=ctx);

    assert greet("Ann") == "hello there";
    cg = ctx.graph;
    assert cg is not None;
    assistant = [
        m
        for m in cg.messages()
        if m.role == "assistant"
    ];
    assert [m.content for m in assistant] == ["hello there"] , (
        f"the reply is recorded exactly once: {[m.content for m in assistant]}"
    );
    assert cg.tool_results() == [];
}


test "persist=True reuses the graph across calls; persist=False makes a new one" {
    kept = Context(persist=True);
    first = MockLLM(outputs=two_lookups());
    run_agent(first, kept);
    cg = kept.graph;
    assert cg is not None;
    before = len(cg.messages());

    second = MockLLM(outputs=[finish("again")]);
    run_agent(second, kept);
    assert kept.graph is cg , "persist=True keeps the same ContextGraph msg_node";
    assert len(cg.messages()) > before , "the second call appends to it";
    assert [tr.call_id for tr in cg.tool_results()] == ["call_a", "call_b"];

    fresh = Context(persist=False);
    run_agent(MockLLM(outputs=[finish("one")]), fresh);
    graph_one = fresh.graph;
    run_agent(MockLLM(outputs=[finish("two")]), fresh);
    graph_two = fresh.graph;
    assert graph_one is not None and graph_two is not None;
    assert graph_one is not graph_two , "persist=False starts a new graph per call";
    assert len(graph_two.messages()) == len(graph_one.messages());

    kept.reset();
    assert kept.graph is None;
    run_agent(MockLLM(outputs=[finish("reset")]), kept);
    assert kept.graph is not None and kept.graph is not cg;
}


test "without context nothing is wired and the request is unchanged" {
    run = mk_run();
    assert run.context is None;
    assert run.context_policy is None;
    assert run._ctx_msg_ids == {};

    plain = MockLLM(outputs=two_lookups());
    wired = MockLLM(outputs=two_lookups());

    assert run_agent(plain, None) == run_agent(wired, Context(policy=ContextPolicy()));
    assert len(plain.seen) == 3 and len(wired.seen) == 3;
    for n in range(3) {
        assert sent_messages(plain, n) == sent_messages(wired, n) , (
            f"call {n}: a record-only Context must not change what the model sees"
        );
    }
    assert plain.sent("tools") == wired.sent("tools");
}


test "the policy overlay reaches the model while the canonical history keeps the full text" {
    llm = MockLLM(outputs=two_lookups());
    history: list[dict[str, any]] = [];
    policy = TrimPolicy(keep=120);
    ctx = Context(policy=policy);

    def agent(task: str) -> str by llm(
        tools=[lookup], context=ctx, conversation=history, max_react_iterations=2
    );

    assert agent("find alpha and beta") == "done";
    assert len(llm.seen) == 3 , (
        f"two tool rounds then the forced final answer, got {len(llm.seen)} calls"
    );
    assert policy.passes == 2 , (
        f"prune runs before every dispatch after the first, got {policy.passes}"
    );

    first = sent_messages(llm, 0);
    second = sent_messages(llm, 1);
    third = sent_messages(llm, 2);
    assert [m["role"] for m in first] == ["system", "user"];

    full_a = lookup("alpha");
    trimmed_a = tool_message(second, "call_a")["content"];
    assert SOFT_TRIM_MARKER.strip() in trimmed_a , f"call_a is sent trimmed: {trimmed_a[:200]!r}";
    assert trimmed_a.startswith(full_a[:120]) and trimmed_a.endswith(full_a[-120:]);
    assert len(trimmed_a) < len(full_a);
    assert tool_message(second, "call_a")["name"] == "lookup";

    for (n, sent) in [(1, second), (2, third)] {
        assistant_ids: list[str] = [];
        for m in sent {
            if m.get("role") == "assistant" {
                assistant_ids.extend(tool_call_ids(m));
            }
        }
        tool_ids = [
            m["tool_call_id"]
            for m in sent
            if m.get("role") == "tool"
        ];
        assert assistant_ids == tool_ids , (
            f"call {n}: every tool_use keeps its tool_result, in order: {assistant_ids} vs {tool_ids}"
        );
        assert [m["role"] for m in sent][:2] == ["system", "user"];
    }
    assert SOFT_TRIM_MARKER.strip() in tool_message(third, "call_b")["content"] , (
        "call_b is trimmed on the pass before the forced answer"
    );
    assert third[-1]["role"] == "user" and "final answer" in third[-1]["content"] , (
        f"the forced-answer scaffolding message passes through untouched: {third[-1]}"
    );
    assert second[1] == first[1] , "the task message is never rewritten";

    cg = ctx.graph;
    assert cg is not None;
    tr_a = cg.tool_results()[0];
    assert tr_a.truncation_state == "head_tail";
    assert tr_a.full_content == full_a , "the graph keeps the full text";
    assert tr_a.visible_content == trimmed_a , "the model saw exactly the graph's visible_content";

    kept_a = [
        m
        for m in history
        if m.get("role") == "tool" and m.get("tool_call_id") == "call_a"
    ];
    assert len(kept_a) == 1 and kept_a[0]["content"] == full_a , (
        "the caller's conversation list keeps the full tool result"
    );
    assert [m["role"] for m in history]
        == ["user", "assistant", "tool", "assistant", "tool"] , (
        f"history holds the run minus scaffolding and finish plumbing: {[
            m['role'] for m in history
        ]}"
    );
}


test "materialize_overlay hides compacted messages and inserts the summary once" {
    ctx = Context(policy=ContextPolicy(), persist=False);
    cg = ctx.get_or_create_graph();
    run = mk_run(tools=[lookup]);
    run.context = cg;
    for m in run.messages {
        msg_node = cg.record_message(role=message_role(m), content=message_text(m));
        run._ctx_msg_ids[id(m)] = msg_node.msg_id;
    }
    def assistant_call(call_id: str, key: str) -> dict[str, any] {
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "lookup", "arguments": f'{{"key": "{key}"}}'}
                }
            ]
        };
    }
    def add_round(
        call_id: str, key: str
    ) -> tuple[dict[str, any], any, ToolCallResultMsg, any] {
        call_msg = assistant_call(call_id, key);
        run.messages.append(call_msg);
        msg_node = cg.record_message(
            role="assistant",
            content="",
            tool_calls=[{"id": call_id, "name": "lookup", "args": {"key": key}}]
        );
        run._ctx_msg_ids[id(call_msg)] = msg_node.msg_id;
        result = ToolCallResultMsg(
            content=lookup(key), tool_call_id=call_id, name="lookup"
        );
        run.messages.append(result);
        tr = cg.record_tool_result(
            tool_name="lookup",
            args={"key": key},
            full_content=lookup(key),
            turn=cg.turn_count,
            call_id=call_id
        );
        return (call_msg, msg_node, result, tr);
    }
    (_old_call, old_node, _old_result, tr_old) = add_round("call_old", "old");
    (new_call, _new_node, new_result, tr_new) = add_round("call_new", "new");
    scaffold = Message(role=MessageRole.USER, content="scaffolding: not on the graph");
    run.messages.append(scaffold);

    # Nothing hidden or trimmed: the overlay is the identity.
    same = materialize_overlay(cg, run, []);
    assert same is not None and same == run.messages;

    # Compact the old round: hide its message and result, add a summary.
    old_node.visible = False;
    tr_old.visible = False;
    cg ++> ContextCompactionSummary(
        covers_turn_start=old_node.turn,
        covers_turn_end=old_node.turn,
        summary_text="old lookup found row_0001",
        visible=True
    );
    apply_eviction_decision(
        EvictionDecision(tool_result=tr_new, kind="hard_clear", placeholder="[cleared]")
    );

    out = materialize_overlay(cg, run, ["[Restored tool result lookup]\nrow_0002"]);
    assert out is not None;
    assert out[0] is run.messages[0] and out[1] is run.messages[1] , "system and first user pass through";
    assert out[2]
        == {
            "role": "user",
            "content": COMPACTION_TAG + "\nold lookup found row_0001"
        } , (f"the summary sits where the first hidden message was: {out[2]}");
    assert out[3] is new_call , "a visible assistant message passes through by identity";
    assert isinstance(out[4], ToolCallResultMsg) and out[4].tool_call_id == "call_new";
    assert out[4].content == "[cleared]" and out[4] is not new_result , "a rendered result is a copy";
    assert new_result.content == lookup("new") , "the canonical message is never mutated";
    assert out[5] is scaffold , "messages with no graph msg_node pass through";
    assert out[6]
        == {"role": "user", "content": "[Restored tool result lookup]\nrow_0002"};
    assert len(out) == 7 , f"hidden round gone, summary and restore added: {len(out)}";
    assert not any(
        isinstance(m, ToolCallResultMsg) and m.tool_call_id == "call_old" for m in out
    );

    # The system message and the first user message are never hidden.
    for m in cg.messages()[:2] {
        m.visible = False;
    }
    guarded = materialize_overlay(cg, run, []);
    assert guarded is not None
        and guarded[0] is run.messages[0]
        and guarded[1] is run.messages[1];
}


test "a logging stream records the same graph as the blocking call" {
    streamed = Context(policy=ContextPolicy());
    llm = MockLLM(
        outputs=[
            call("lookup", {"key": "alpha"}, call_id="call_a"),
            finish("done"),
            "done"
        ]
    );

    def agent(task: str) -> str by llm(
        tools=[lookup], context=streamed, stream=True, logging=True
    );

    events = stream_events(agent("find alpha"));
    assert [e.data["call_id"] for e in events_of(events, "tool_result")] == ["call_a"];

    blocking = Context(policy=ContextPolicy());
    run_agent(
        MockLLM(
            outputs=[call("lookup", {"key": "alpha"}, call_id="call_a"), finish("done")]
        ),
        blocking
    );

    sg = streamed.graph;
    bg = blocking.graph;
    assert sg is not None and bg is not None;
    assert [(tr.call_id, tr.tool_name, tr.full_content) for tr in sg.tool_results()]
        == [(tr.call_id, tr.tool_name, tr.full_content) for tr in bg.tool_results()];
    assert sg.tool_results()[0].produced_at_turn
        == [
            m.turn
            for m in sg.messages()
            if m.role == "assistant"
        ][0]
            + 1;
    stream_roles = [m.role for m in sg.messages()];
    block_roles = [m.role for m in bg.messages()];
    assert stream_roles[:len(block_roles)] == block_roles , (
        f"same recording up to the finish: {stream_roles} vs {block_roles}"
    );
    assert stream_roles[len(block_roles):] in [[], ["assistant"]] , (
        f"the streamed final answer may add one assistant message: {stream_roles}"
    );
}


test "stats() reports counters and the visible token view" {
    policy = TrimPolicy(keep=120);
    ctx = Context(policy=policy, budget_tokens=500);
    keys = [
        "session_id",
        "budget_tokens",
        "prunes",
        "restores",
        "trims",
        "stubs",
        "clears",
        "compactions",
        "turn_count",
        "messages",
        "hidden_messages",
        "tool_results",
        "visible_tool_results",
        "hidden_tool_results",
        "full",
        "trimmed",
        "hard_cleared",
        "summaries",
        "visible_tokens",
        "full_tokens",
        "saved_tokens"
    ];

    empty = ctx.stats();
    for key in keys {
        assert key in empty , f"{key} is reported before any call";
    }
    assert empty["tool_results"] == 0
        and empty["visible_tokens"] == 0
        and empty["turn_count"] == 0;
    assert empty["budget_tokens"] == 500;

    llm = MockLLM(outputs=two_lookups());
    assert run_agent(llm, ctx) == "done";
    s = ctx.stats();
    cg = ctx.graph;
    assert cg is not None;
    assert s["tool_results"] == 2
        and s["visible_tool_results"] == 2
        and s["hidden_tool_results"] == 0;
    assert policy.passes == 2 , f"a pass before each dispatch after the first: {policy.passes}";
    assert s["trimmed"] == 2 and s["full"] == 0 and s["hard_cleared"] == 0 , (
        f"call_a was trimmed on pass 2 and call_b on pass 3: {s}"
    );
    assert s["full_tokens"]
        == estimate_tokens(lookup("alpha")) + estimate_tokens(lookup("beta"));
    assert s["visible_tokens"]
        == sum([estimate_tokens(tr.visible_content) for tr in cg.tool_results()]);
    assert s["saved_tokens"] == s["full_tokens"] - s["visible_tokens"];
    assert s["saved_tokens"] > 0;
    assert s["turn_count"] == len(cg.messages()) + len(cg.tool_results())
        and s["messages"] == len(cg.messages());
    assert s["hidden_messages"] == 0 and s["summaries"] == 0;
    assert s["prunes"] == 0 , "counters come from cg.stats, which this policy does not touch";

    cg.stats["prunes"] = 2;
    cg.stats["trims"] = 1;
    again = ctx.stats();
    assert again["prunes"] == 2 and again["trims"] == 1 and again["stubs"] == 0;
}


test "a summary from an earlier call is sent after the task when its turns are not in this call" {
    # persist=True: the graph outlives one by-llm call. A later call has a fresh
    # message list, so none of the compacted messages are present to anchor the
    # summary; it must still reach the model, right after the task message.
    ctx = Context(policy=ContextPolicy(), persist=False);
    cg = ctx.get_or_create_graph();
    earlier = cg.record_message(role="user", content="an earlier task");
    earlier.visible = False;
    cg ++> ContextCompactionSummary(
        covers_turn_start=earlier.turn,
        covers_turn_end=earlier.turn,
        summary_text="earlier: PORT = 8080 (a.py)",
        visible=True
    );
    run = mk_run(tools=[lookup]);
    run.context = cg;
    for m in run.messages {
        msg_node = cg.record_message(role=message_role(m), content=message_text(m));
        run._ctx_msg_ids[id(m)] = msg_node.msg_id;
    }
    reply = Message(role=MessageRole.ASSISTANT, content="working on it");
    run.messages.append(reply);

    out = materialize_overlay(cg, run, []);
    assert out is not None;
    assert out[0] is run.messages[0] and out[1] is run.messages[1] , "system and task first";
    assert out[2]
        == {
            "role": "user",
            "content": COMPACTION_TAG + "\nearlier: PORT = 8080 (a.py)"
        } , (f"the earlier summary follows the task: {out[2]}");
    assert out[3] is reply and len(out) == 4 , f"nothing else changes: {len(out)}";
}


test "persist=True does not re-record the history a later call sends again" {
    # A persisted graph sees the same system prompt on every call, and with
    # conversation= it sees the whole history again. Messages that match the
    # recorded prefix in order are mapped to their existing nodes, not recorded twice.
    llm = MockLLM(outputs=[say("one"), say("two")]);
    ctx = Context(policy=ContextPolicy(), persist=True);
    def ask(q: str) -> str by llm(context=ctx);
    ask("first");
    cg = ctx.graph;
    assert cg is not None;
    after_one = len(cg.messages());
    systems_one = len(
        [
            m
            for m in cg.messages()
            if m.role == "system"
        ]
    );
    ask("second");
    assert ctx.graph is cg;
    systems_two = len(
        [
            m
            for m in cg.messages()
            if m.role == "system"
        ]
    );
    assert systems_two == systems_one , f"system prompt recorded again: {systems_one} -> {systems_two}";
    added = len(cg.messages()) - after_one;
    assert added == 2 , f"the second call adds its user and assistant messages only, got {added}";
}


test "results with an empty or a duplicate call id are recorded but never managed" {
    # The overlay pairs a rendering with its tool message by call id. Without a
    # unique id that pairing is ambiguous, so such results stay out of the
    # managed set and their messages pass through untouched.
    ctx = Context(policy=ContextPolicy(), persist=False);
    cg = ctx.get_or_create_graph();
    cg.record_message(role="user", content="task");
    ok = cg.record_tool_result(
        tool_name="lookup", args={}, full_content=lookup("ok"), call_id="call_ok"
    );
    dup_a = cg.record_tool_result(
        tool_name="lookup", args={}, full_content=lookup("dup_a"), call_id="call_dup"
    );
    dup_b = cg.record_tool_result(
        tool_name="lookup", args={}, full_content=lookup("dup_b"), call_id="call_dup"
    );
    anon = cg.record_tool_result(
        tool_name="lookup", args={}, full_content=lookup("anon"), call_id=""
    );
    assert [tr.produced_at_turn for tr in (ok, dup_a, dup_b, anon)] == [2, 3, 4, 5] , "each result takes its own turn";
    assert [tr.call_id for tr in managed_tool_results(cg)] == ["call_ok"];

    for tr in (ok, dup_a, dup_b, anon) {
        apply_eviction_decision(
            EvictionDecision(tool_result=tr, kind="hard_clear", placeholder="[cleared]")
        );
    }
    run = mk_run(tools=[lookup]);
    run.context = cg;
    msg_ok = ToolCallResultMsg(
        content=lookup("ok"), tool_call_id="call_ok", name="lookup"
    );
    msg_dup_a = ToolCallResultMsg(
        content=lookup("dup_a"), tool_call_id="call_dup", name="lookup"
    );
    msg_dup_b = ToolCallResultMsg(
        content=lookup("dup_b"), tool_call_id="call_dup", name="lookup"
    );
    msg_anon = ToolCallResultMsg(
        content=lookup("anon"), tool_call_id="", name="lookup"
    );
    run.messages.extend([msg_ok, msg_dup_a, msg_dup_b, msg_anon]);
    out = materialize_overlay(cg, run, []);
    assert out is not None;
    assert out[2].content == "[cleared]" and out[2].tool_call_id == "call_ok" , "the managed result is rendered";
    assert out[3] is msg_dup_a and out[4] is msg_dup_b and out[5] is msg_anon , "unmanaged results pass through by identity";
}


"""An assistant message carrying one tool call, as a provider returns it."""
def assistant_call(call_id: str, name: str, args: str = "{}") -> dict[str, any] {
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args}
            }
        ]
    };
}


"""Fail unless every tool message follows an assistant message that made its call,
and every tool call an assistant message makes has its tool message."""
def assert_paired(messages: list[dict[str, any]]) {
    called: list[str] = [];
    answered: list[str] = [];
    for m in messages {
        if m.get("role") == "assistant" {
            called.extend(tool_call_ids(m));
        } elif m.get("role") == "tool" {
            call_id = str(m.get("tool_call_id", ""));
            assert call_id in called , f"tool message {call_id} has no tool call before it";
            answered.append(call_id);
        }
    }
    assert sorted(called) == sorted(answered) , f"calls {called} but results {answered}";
}


test "message_text reads a block list the same before and after a conversation round trip" {
    picture = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}};
    blocks: list[any] = [
        Text(text="first line\nsecond line"),
        picture,
        picture,
        Text(text="tail")
    ];
    msg = Message(role=MessageRole.USER, content=blocks);
    want = "first line\nsecond line\n[media]\ntail";
    assert message_text(msg) == want , f"got {message_text(msg)!r}";
    as_dict = msg.to_dict();
    assert message_text(as_dict) == want , f"got {message_text(as_dict)!r}";
    assert message_text(Message.from_dict(as_dict)) == want , "the round trip reads the same";
    assert message_text({"role": "assistant", "content": None}) == "";
    assert message_text({"role": "user", "content": "plain"}) == "plain";
}


test "conversation= history maps onto the persisted graph and stays paired after a fold" {
    history: list[dict[str, any]] = [];
    llm = MockLLM(
        outputs=[
            call("lookup", {"key": "alpha"}, call_id="call_a"),
            finish("one"),
            call("lookup", {"key": "beta"}, call_id="call_b"),
            finish("two"),
            finish("three"),
            finish("four")
        ]
    );
    ctx = Context(policy=ViewPolicy(), persist=True);

    def ask(q: str) -> str by llm(tools=[lookup], conversation=history, context=ctx);

    ask("first");
    cg = ctx.graph;
    assert cg is not None;
    counts = [len(cg.messages())];
    ask("second");
    counts.append(len(cg.messages()));
    ask("third");
    counts.append(len(cg.messages()));
    # Each call adds its own task and its own assistant turns, never the history again:
    # system + task + call + finish, then task + call + finish, then task + finish.
    assert counts == [4, 7, 9] , f"history was recorded again: {counts}";
    assert [m.role for m in cg.messages()].count("system") == 1;
    assert [tr.call_id for tr in managed_tool_results(cg)] == ["call_a", "call_b"];

    # Fold the first round the way a compaction does, then send the history once more.
    round_a = [
        m
        for m in cg.messages()
        if [tc["id"] for tc in m.tool_calls] == ["call_a"]
    ];
    assert len(round_a) == 1 , "the first tool call has exactly one node";
    round_a[0].visible = False;
    [
        tr
        for tr in cg.tool_results()
        if tr.call_id == "call_a"
    ][0].visible = False;
    cg ++> ContextCompactionSummary(
        covers_turn_start=round_a[0].turn,
        covers_turn_end=round_a[0].turn + 1,
        summary_text="alpha was looked up",
        visible=True
    );
    assert ask("fourth") == "four";
    assert len(cg.messages()) == 11 , f"task and finish only: {len(cg.messages())}";
    sent = sent_messages(llm, 5);
    assert_paired(sent);
    assert "call_a" not in str(sent) , "the folded round is not sent in any form";
    assert tool_message(sent, "call_b")["content"] == lookup("beta");
    assert [
        m
        for m in sent
        if str(m.get("content", "")).startswith(COMPACTION_TAG)
    ]
        == [{"role": "user", "content": COMPACTION_TAG + "\nalpha was looked up"}];
    assert [
        m
        for m in history
        if m.get("tool_call_id") == "call_a"
    ][0]["content"]
        == lookup("alpha") , ("the caller's history keeps the folded result");
}


test "tool results in conversation= history are recorded and managed on a fresh graph" {
    history: list[dict[str, any]] = [
        {"role": "user", "content": "an earlier task"},
        assistant_call("call_h", "lookup", '{"key": "hist"}'),
        {
            "role": "tool",
            "content": lookup("hist"),
            "tool_call_id": "call_h",
            "name": "lookup"
        },
        {"role": "assistant", "content": "an earlier answer"}
    ];
    llm = MockLLM(
        outputs=[call("lookup", {"key": "alpha"}, call_id="call_a"), finish("done")]
    );
    ctx = Context(policy=TrimPolicy(), persist=False);

    def agent(task: str) -> str by llm(
        tools=[lookup], conversation=history, context=ctx
    );

    assert agent("now alpha") == "done";
    cg = ctx.graph;
    assert cg is not None;
    by_id = {tr.call_id: tr for tr in managed_tool_results(cg)};
    assert sorted(by_id.keys()) == ["call_a", "call_h"] , sorted(by_id.keys());
    hist = by_id["call_h"];
    assert hist.tool_name == "lookup" and hist.args == {"key": "hist"} , (
        hist.tool_name,
        hist.args
    );
    assert hist.full_content == lookup("hist");
    assert hist.produced_at_turn < by_id["call_a"].produced_at_turn , "history comes first";

    second = sent_messages(llm, 1);
    assert_paired(second);
    assert SOFT_TRIM_MARKER in tool_message(second, "call_h")["content"] , (
        "a result that arrived as history is trimmed like any other"
    );
    assert [
        m
        for m in history
        if m.get("tool_call_id") == "call_h"
    ][0]["content"]
        == lookup("hist") , ("the caller's history keeps the full text");
}


test "the overlay keeps a tool call and its result together when one side is hidden" {
    ctx = Context(policy=ContextPolicy(), persist=False);
    cg = ctx.get_or_create_graph();
    run = mk_run(tools=[lookup]);
    run.context = cg;
    for m in run.messages {
        msg_node = cg.record_message(role=message_role(m), content=message_text(m));
        run._ctx_msg_ids[id(m)] = msg_node.msg_id;
    }

    # A result with no node: a finish_tool validation error is added this way.
    err_call = assistant_call("call_err", "finish_tool");
    run.messages.append(err_call);
    err_node = cg.record_message(
        role="assistant",
        content="",
        tool_calls=[{"id": "call_err", "name": "finish_tool", "args": {}}]
    );
    run._ctx_msg_ids[id(err_call)] = err_node.msg_id;
    err_result = ToolCallResultMsg(
        content="not a valid answer", tool_call_id="call_err", name="finish_tool"
    );
    run.messages.append(err_result);

    # A result with a node whose assistant message has none.
    kept_call = assistant_call("call_kept", "lookup");
    run.messages.append(kept_call);
    kept_result = ToolCallResultMsg(
        content=lookup("kept"), tool_call_id="call_kept", name="lookup"
    );
    run.messages.append(kept_result);
    tr_kept = cg.record_tool_result(
        tool_name="lookup", args={}, full_content=lookup("kept"), call_id="call_kept"
    );

    both = materialize_overlay(cg, run, []);
    assert both is not None and both == run.messages , "nothing hidden: the identity";

    err_node.visible = False;
    tr_kept.visible = False;
    out = materialize_overlay(cg, run, []);
    assert out is not None;
    assert not any(m is err_call or m is err_result for m in out) , (
        "a hidden tool call takes its result with it, managed or not"
    );
    assert not any(tool_call_id_of(m) == "call_err" for m in out);
    assert any(m is kept_call for m in out) , "a message with no node is always sent";
    folded = [
        m
        for m in out
        if tool_call_id_of(m) == "call_kept"
    ];
    assert len(folded) == 1 and folded[0].content == FOLDED_RESULT_NOTE , (
        f"a sent tool call keeps a result message: {folded}"
    );
    assert kept_result.content == lookup("kept") , "the canonical message is never mutated";
}


test "a policy that does not compact leaves auto-compaction to the built-in summary" {
    llm = MockLLM(
        ctx_window=1000,
        outputs=[
            call(
                "lookup",
                {"key": "alpha"},
                call_id="call_a",
                usage={"prompt_tokens": 100}
            ),
            call(
                "lookup",
                {"key": "beta"},
                call_id="call_b",
                usage={"prompt_tokens": 850}
            ),
            say("alpha was looked up"),
            finish("ok")
        ]
    );
    ctx = Context(policy=ContextPolicy(), persist=False);

    def agent(task: str) -> str by llm(
        tools=[lookup], context=ctx, threshold_ratio=0.8, keep_recent_iterations=1
    );

    assert agent("find alpha and beta") == "ok";
    assert len(llm.seen) == 4 , f"two rounds, the summary call, the finish: {len(
        llm.seen
    )}";
    assert llm.exhausted();
    after = sent_messages(llm, 3);
    assert_paired(after);
    assert after[2]
        == {"role": "user", "content": COMPACTION_TAG + "\nalpha was looked up"} , after[
        2
    ];
    assert [
        m.get("tool_call_id")
        for m in after
        if m.get("role") == "tool"
    ]
        == ["call_b"];
    assert ctx.stats()["compactions"] == 0 , "the policy itself folded nothing";
}
