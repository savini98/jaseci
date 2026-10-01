"""ContextRender: the scoring, selection and rendering policy behind
`by llm(context=Context(...))`.

Budgets are estimated tokens (chars // 4), so every tool result here has a known
character count. Tests that need exact budget arithmetic build the graph by hand,
recording each round the way the ReAct loop does (the assistant message and its
tool message land in `run.messages`, every item takes a turn), and call the policy
directly; tests whose subject is the wiring go through the loop with a MockLLM.
"""

import json;
import math;
import time;
import litellm;
import jaclang.byllm.context_render as render_module;
import from unittest { mock }
import from jaclang.byllm.lib { MockLLM }
import from jaclang.byllm.mtir { MTRuntime }
import from jaclang.byllm.context {
    COMPACTION_TAG,
    Context,
    ContextGraph,
    ContextMessage,
    ContextToolResult,
    EvictionDecision,
    ReferencedBy,
    apply_eviction_decision,
    cached_lextoks,
    estimate_tokens,
    lexical_set_cosine,
    lexical_tokens,
    message_role,
    message_text
}
import from jaclang.byllm.context_render {
    ContextRender,
    EMBED_CACHE,
    EMBED_STATE,
    HARD_CLEAR_PLACEHOLDER,
    TRIM_MARKER,
    active_call_ids,
    focus_texts,
    presence_of,
    scan_citations,
    visible_tool_tokens
}
import from jaclang.byllm.exceptions { CompactionNotEffectiveError }
import from jaclang.byllm.types { ToolCallResultMsg }
import from jaclang.byllm.typed_compaction { CompactionSlots }
import from support { call, calls, fail, finish, mk_run, roles, sent_messages }


glob VALUE_LINE: str = "keepalive_seconds = 47 (config/prod.yaml)";


"""Stands in for the byLLM config: only `[plugins.byllm.context]` is read."""
obj FakeConfig {
    has context: dict[str, any];

    def get_context_config -> dict[str, any] {
        return self.context;
    }
}


"""`n` lines whose every token carries `tag`, so two different tags share nothing."""
def rows(tag: str, n: int) -> str {
    lines: list[str] = [];
    for i in range(n) {
        lines.append(f"{tag}_row_{str(i).zfill(4)} {tag}_val_{str(i * 7).zfill(6)}");
    }
    return "\n".join(lines);
}


"""Exactly `chars` characters of `rows(tag, ...)`, so its token estimate is chars // 4."""
def blob(tag: str, chars: int) -> str {
    return rows(tag, chars // 10 + 2)[:chars];
}


"""A tool whose result size the script chooses."""
def fetch(key: str, chars: int) -> str {
    return blob(key, chars);
}


"""The content of the tool message for `call_id` in one sent conversation."""
def tool_text(messages: list[dict[str, any]], call_id: str) -> str {
    found = [
        m
        for m in messages
        if m.get("role") == "tool" and m.get("tool_call_id") == call_id
    ];
    assert len(found) == 1 , f"expected one tool message for {call_id}, got {len(
        found
    )}";
    return str(found[0]["content"]);
}


def has_tool_message(messages: list[dict[str, any]], call_id: str) -> bool {
    return any(
        m.get("role") == "tool" and m.get("tool_call_id") == call_id for m in messages
    );
}


"""A run with the system and user messages recorded into `ctx`'s graph and carrying
`ctx` in its call params, exactly as MTRuntime.factory wires it."""
def wired(ctx: Context, task: str = "") -> tuple[ContextGraph, MTRuntime] {
    run = mk_run(tools=[fetch], call_params={"context": ctx});
    if task {
        run.messages[1] = {"role": "user", "content": task};
    }
    cg = ctx.get_or_create_graph();
    run.context = cg;
    run.context_policy = ctx.policy;
    for m in run.messages {
        msg_node = cg.record_message(role=message_role(m), content=message_text(m));
        run._ctx_msg_ids[id(m)] = msg_node.msg_id;
    }
    return (cg, run);
}


"""Record a user or assistant message the way the loop does: into the run and the graph."""
def add_message(
    cg: ContextGraph, run: MTRuntime, role: str, text: str
) -> ContextMessage {
    m: dict[str, any] = {"role": role, "content": text};
    run.messages.append(m);
    msg_node = cg.record_message(role=role, content=text);
    run._ctx_msg_ids[id(m)] = msg_node.msg_id;
    return msg_node;
}


"""Record one assistant round the way the loop does: the tool call (with optional
prose) and its result, each taking its own turn; returns the result node."""
def add_result(
    cg: ContextGraph,
    run: MTRuntime,
    call_id: str,
    content: str,
    args: dict[str, any] | None = None,
    text: str = "",
    tool_name: str = "fetch"
) -> ContextToolResult {
    a: dict[str, any] = args if args is not None else {"key": "x"};
    call_msg: dict[str, any] = {
        "role": "assistant",
        "content": text or None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(a)}
            }
        ]
    };
    run.messages.append(call_msg);
    msg_node = cg.record_message(
        role="assistant",
        content=text,
        tool_calls=[{"id": call_id, "name": tool_name, "args": a}]
    );
    run._ctx_msg_ids[id(call_msg)] = msg_node.msg_id;
    run.messages.append(
        ToolCallResultMsg(content=content, tool_call_id=call_id, name=tool_name)
    );
    return cg.record_tool_result(
        tool_name=tool_name, args=a, full_content=content, call_id=call_id
    );
}


def by_call(cg: ContextGraph) -> dict[str, ContextToolResult] {
    return {tr.call_id: tr for tr in cg.tool_results()};
}


def states(cg: ContextGraph) -> dict[str, str] {
    return {tr.call_id: tr.truncation_state for tr in cg.tool_results()};
}


"""Visible tokens of every result on the graph, managed or not."""
def visible_tokens(cg: ContextGraph) -> int {
    return sum(
        [
            estimate_tokens(tr.visible_content)
            for tr in cg.tool_results()
            if tr.visible
        ]
    );
}


"""The stub's own lines, without the elision markers."""
def stub_lines(text: str) -> list[str] {
    return [
        line
        for line in text.split("\n")
        if line != TRIM_MARKER
    ];
}


# 6. Budget pressure through the loop.
test "under budget pressure the oldest results are trimmed, the newest stays full, nothing is hard-cleared" {
    llm = MockLLM(
        outputs=[
            call("fetch", {"key": "alpha", "chars": 8000}, call_id="call_a"),
            call("fetch", {"key": "beta", "chars": 8000}, call_id="call_b"),
            call("fetch", {"key": "gamma", "chars": 8000}, call_id="call_c"),
            finish("done")
        ]
    );
    ctx = Context(budget_tokens=3600, persist=False);
    assert isinstance(ctx.policy, ContextRender) , "Context() defaults to ContextRender";

    def agent(task: str) -> str by llm(tools=[fetch], context=ctx);

    assert agent("collect alpha, beta and gamma") == "done";
    assert len(llm.seen) == 4;

    # Call 2 saw both results at full fidelity: 4000 tokens do not exceed 3600 until beta lands.
    assert tool_text(sent_messages(llm, 1), "call_a") == blob("alpha", 8000);
    # The pass before call 3 trimmed alpha (2000 + 2000 > 3600); beta was the newest.
    third = sent_messages(llm, 2);
    assert TRIM_MARKER in tool_text(third, "call_a") , "alpha is sent trimmed on call 3";
    assert tool_text(third, "call_b") == blob("beta", 8000) , "beta, the newest, stays full";

    last = sent_messages(llm, 3);
    a = tool_text(last, "call_a");
    b = tool_text(last, "call_b");
    c = tool_text(last, "call_c");
    assert TRIM_MARKER in a and TRIM_MARKER in b , "the two oldest are trimmed on the last call";
    assert c == blob("gamma", 8000) , "the newest result is full (grace window)";
    assert a.startswith(blob("alpha", 8000)[:900])
        and a.endswith(blob("alpha", 8000)[-900:]) , (
        "a smart trim keeps a head and a tail anchor"
    );
    assert len(a) < 8000 and len(b) < 8000;
    assert HARD_CLEAR_PLACEHOLDER not in str(last)
        and "evicted from context" not in str(last);
    assert [m["role"] for m in last][:2] == ["system", "user"];

    cg = ctx.graph;
    assert cg is not None;
    assert states(cg)
        == {"call_a": "head_tail", "call_b": "head_tail", "call_c": "full"} , states(
        cg
    );
    assert visible_tokens(cg) <= 3600 , f"visible tool tokens fit the budget: {visible_tokens(
        cg
    )}";
    assert a == by_call(cg)["call_a"].visible_content , "the model saw the graph's visible_content";
    turns = [tr.produced_at_turn for tr in cg.tool_results()];
    asst_turns = [
        m.turn
        for m in cg.messages()
        if m.role == "assistant"
    ];
    assert turns == [asst_turns[0] + 1, asst_turns[1] + 1, asst_turns[2] + 1] , (
        f"every item is a turn: results sit one after their call: {turns} vs {asst_turns}"
    );
    s = ctx.stats();
    assert s["budget_tokens"] == 3600 and s["prunes"] == 2 and s["trims"] == 2 , s;
    assert s["clears"] == 0 and s["hard_cleared"] == 0 and s["stubs"] == 0 , s;
    assert s["saved_tokens"] > 0;
}


# 7. The score formula.
test "base_score is decayed recency plus saturating reuse plus clipped relevance" {
    policy = ContextRender();
    for (label, age, cmass, rel) in [
        ("fresh, unused, irrelevant", 0, 0.0, 0.0),
        ("three turns old with one citation", 3, 1.0, 0.25),
        ("negative relevance is clipped to zero", 2, 0.5, -0.4),
        ("heavy reuse saturates", 10, 40.0, 1.0)
    ] {
        want = (
            math.exp(-0.3 * age)
            + (1.0 - math.exp(-0.2 * cmass))
            + (rel if rel > 0.0 else 0.0)
        );
        got = policy.base_score(age, cmass, rel);
        assert abs(got - want) < 1e-12 , f"{label}: {got} vs {want}";
    }
    assert policy.base_score(0, 0.0, 0.0) == 1.0;
    assert abs(policy.base_score(2, 0.5, -0.4) - policy.base_score(2, 0.5, 0.0))
        < 1e-12;

    weighted = ContextRender(
        w_recency=2.0,
        w_reuse=0.5,
        w_relevance=3.0,
        recency_decay=0.1,
        reuse_saturation=1.0
    );
    want = 2.0 * math.exp(-0.1 * 4) + 0.5 * (1.0 - math.exp(-1.0 * 2.0)) + 3.0 * 0.5;
    assert abs(weighted.base_score(4, 2.0, 0.5) - want) < 1e-12;

    capped = ContextRender(reuse_cap=1.0);
    assert capped.base_score(0, 5.0, 0.0) == capped.base_score(0, 1.0, 0.0) , "reuse_cap clips the mass";
    assert ContextRender().base_score(0, 5.0, 0.0)
        > ContextRender().base_score(0, 1.0, 0.0) , ("off by default (-1)");

    # On a graph: ages come from turns (every item is one), mass from citation_counts,
    # relevance from the lexical query (anchor and focus tokens).
    ctx = Context(policy=policy, budget_tokens=10000, persist=False);
    (cg, run) = wired(ctx);
    tr_a = add_result(
        cg, run, "call_a", "the thing is alpha_row_0001\n" + blob("alpha", 300)
    );
    tr_b = add_result(cg, run, "call_b", blob("beta", 300));
    cg.citation_counts["call_a"] = 2.0;
    query = lexical_tokens("Do the thing.") | lexical_tokens("Do the thing.");
    rel_a = lexical_set_cosine(cached_lextoks(tr_a.full_content), query);
    rel_b = lexical_set_cosine(cached_lextoks(tr_b.full_content), query);
    assert rel_a > 0.0 and rel_b == 0.0 , f"alpha shares 'the thing' with the task: {rel_a}, {rel_b}";
    scores = policy.candidate_scores(cg, run);
    age_a = cg.turn_count - tr_a.produced_at_turn;
    age_b = cg.turn_count - tr_b.produced_at_turn;
    assert (age_a, age_b) == (2, 0) , f"alpha is two items old (beta's call and result): {(
        age_a,
        age_b
    )}";
    assert abs(scores[tr_a.tr_id] - policy.base_score(2, 2.0, rel_a)) < 1e-12 , scores;
    assert abs(scores[tr_b.tr_id] - policy.base_score(0, 0.0, 0.0)) < 1e-12 , scores;
    assert abs(scores[tr_a.tr_id] - (math.exp(-0.6) + (1.0 - math.exp(-0.4)) + rel_a))
        < 1e-12;
}


# 8. The live citation scan.
test "a citation credits only the earliest introducer, weighted 1/df, never a later result, never prose" {
    ctx = Context(budget_tokens=10000, persist=False);
    (cg, run) = wired(ctx);
    tr_a = add_result(
        cg,
        run,
        "call_a",
        "CFG_TIMEOUT_MS=30000\nRETRY_LIMIT_X=5\nplain words only here"
    );
    tr_b = add_result(
        cg, run, "call_b", "CFG_TIMEOUT_MS=30000 seen again\nTimeout Timeout"
    );
    citing = add_message(
        cg,
        run,
        "assistant",
        "Set CFG_TIMEOUT_MS to 30000 and keep RETRY_LIMIT_X. Timeout handled."
    );
    cite_turn = citing.turn;
    tr_c = add_result(
        cg, run, "call_c", "CFG_TIMEOUT_MS=30000 RETRY_LIMIT_X=5 copied later"
    );

    scan_citations(cg);

    assert abs(cg.citation_counts.get("call_a", 0.0) - 1.5) < 1e-12 , (
        f"CFG_TIMEOUT_MS is shared by A and B (1/2, to the earliest) and RETRY_LIMIT_X is A's alone (1): "
        f"{cg.citation_counts}"
    );
    assert cg.citation_counts.get("call_b", 0.0) == 0.0 , (
        f"B introduced nothing first, and 'Timeout' is Titlecase prose: {cg.citation_counts}"
    );
    assert "call_c" not in cg.citation_counts , "a result produced after the citing turn is never credited";
    assert cg.citation_turns.get("call_a") == [cite_turn] , cg.citation_turns;
    assert "call_b" not in cg.citation_turns;
    assert set(cg.citation_idents["call_a"]) == {"CFG_TIMEOUT_MS", "RETRY_LIMIT_X"} , cg.citation_idents[
        "call_a"
    ];
    assert cg.n_seen_asst == 4 , f"every assistant message is now behind the cursor: {cg.n_seen_asst}";

    # The reserved schema is written too: one ReferencedBy edge per (citing message, source),
    # reference_count per distinct citing turn, and a last_referenced_at stamp.
    targets = [citing->:ReferencedBy:->];
    assert [t.call_id for t in targets] == ["call_a"] , f"one edge to the credited source: {targets}";
    edges = [edge citing->:ReferencedBy:->];
    assert [e.cite_turn for e in edges] == [cite_turn];
    assert tr_a.reference_count == 1 == len(cg.citation_turns["call_a"]);
    assert tr_a.last_referenced_at > 0.0;
    assert tr_b.reference_count == 0 and tr_b.last_referenced_at == 0.0;
    assert tr_c.reference_count == 0;

    scan_citations(cg);
    assert abs(cg.citation_counts["call_a"] - 1.5) < 1e-12 , "rescanning credits nothing twice";
    assert cg.citation_turns["call_a"] == [cite_turn];
    assert len([edge citing->:ReferencedBy:->]) == 1 and tr_a.reference_count == 1;

    # A tool call's arguments are scanned too, and a second citing turn is appended.
    add_result(cg, run, "call_d", blob("delta", 100), args={"key": "RETRY_LIMIT_X"});
    d_call = [
        m
        for m in cg.messages()
        if m.role == "assistant"
    ][-1];
    scan_citations(cg);
    assert abs(cg.citation_counts["call_a"] - (1.5 + 0.5)) < 1e-12 , (
        f"RETRY_LIMIT_X now has two earlier sources (A and C), so A gets 1/2: {cg.citation_counts}"
    );
    assert cg.citation_turns["call_a"] == [cite_turn, d_call.turn] , cg.citation_turns;
    assert tr_a.reference_count == 2 == len(cg.citation_turns["call_a"]);
    assert [t.call_id for t in [d_call->:ReferencedBy:->]] == ["call_a"];
    assert len([edge citing->:ReferencedBy:->]) == 1 , "the earlier message's edge is not duplicated";
}


# 9. The working-set pin.
test "a trimmed result whose identifiers the last tool call uses is pinned back to full" {
    for (label, key, want_active, want_state) in [
        ("the call names alpha_row_0007", "alpha_row_0007", {"call_a": True}, "full"),
        ("the call names nothing alpha holds", "nothing_here_9", {}, "head_tail")
    ] {
        policy = ContextRender(stub_overflow="clear", pin_recent=0, w_reuse=0.0);
        ctx = Context(policy=policy, budget_tokens=950, persist=False);
        (cg, run) = wired(ctx);
        add_result(cg, run, "call_a", blob("alpha", 800));
        add_result(cg, run, "call_x", blob("xray", 2000));
        add_result(cg, run, "call_c", blob("cee", 400));
        add_result(cg, run, "call_d", blob("dee", 400));
        add_result(cg, run, "call_e", blob("eee", 400));

        assert policy.should_prune(cg, run, MockLLM()) , f"{label}: 1000 tokens exceed 950";
        policy.prune(cg, run, MockLLM());
        assert states(cg)
            == {
                "call_a": "head_tail",
                "call_x": "head_tail",
                "call_c": "full",
                "call_d": "full",
                "call_e": "full"
            } , f"{label}: first pass stubs the two old results: {states(cg)}";
        assert "alpha_row_0007" in by_call(cg)["call_a"].full_content;

        add_result(cg, run, "call_f", blob("eff", 400), args={"key": key});
        assert active_call_ids(cg, 3) == want_active , f"{label}: {active_call_ids(
            cg, 3
        )}";
        policy.prune(cg, run, MockLLM());
        assert states(cg)["call_a"] == want_state , f"{label}: {states(cg)}";
        assert states(cg)["call_x"] == "head_tail" , f"{label}: the other old result stays stubbed";
        assert states(cg)["call_f"] == "full";
        if want_state == "full" {
            assert by_call(cg)["call_a"].visible_content == blob("alpha", 800);
            assert ctx.stats()["restores"] == 1 , ctx.stats();
        } else {
            assert ctx.stats()["restores"] == 0 , ctx.stats();
        }
    }
}


# 10. Restore on reselect; a duplicate-id result is neither counted nor touched.
test "a trimmed result returns to full when a larger budget reselects it" {
    policy = ContextRender();
    small = Context(policy=policy, budget_tokens=900, persist=False);
    (cg, run) = wired(small);
    # Two results share one id (MockLLM's unnamed calls reuse call_1 every round):
    # the model can never see them rewritten, so they are never budgeted or stubbed.
    add_result(cg, run, "call_dup", blob("dupa", 4000));
    add_result(cg, run, "call_dup", blob("dupb", 4000));
    add_result(cg, run, "call_a", blob("alpha", 2000));
    add_result(cg, run, "call_b", blob("beta", 2000));
    add_result(cg, run, "call_c", blob("cee", 2000));
    present = presence_of(run);
    assert visible_tool_tokens(cg, present) == 1500 , (
        f"the duplicates' 2000 tokens are not counted: {visible_tool_tokens(
            cg, present
        )}"
    );
    assert visible_tokens(cg) == 3500 , "though they are on the graph";

    assert policy.should_prune(cg, run, MockLLM());
    changed = policy.prune(cg, run, MockLLM());
    assert changed == 2 , f"two results changed rendering: {changed}";
    assert states(cg)
        == {
            "call_dup": "full",
            "call_a": "head_tail",
            "call_b": "head_tail",
            "call_c": "full"
        } , states(cg);
    dups = [
        tr
        for tr in cg.tool_results()
        if tr.call_id == "call_dup"
    ];
    assert [tr.truncation_state for tr in dups] == ["full", "full"] , "duplicates are never stubbed";
    assert [tr.visible_content == tr.full_content for tr in dups] == [True, True];
    assert visible_tool_tokens(cg, present) <= 900;
    assert small.stats()["stubs"] == 2 and small.stats()["restores"] == 0;

    # The same graph under a Context with a larger budget: nothing is over budget any more, so the
    # high-water trigger would not fire; a pass that does run (this one) reselects and restores.
    bigger = Context(policy=policy, budget_tokens=100000, persist=False);
    bigger.graph = cg;
    run2 = mk_run(tools=[fetch], call_params={"context": bigger});
    run2.messages = run.messages;
    run2._ctx_msg_ids = run._ctx_msg_ids;
    run2.context = cg;
    run2.context_policy = policy;
    assert not policy.should_prune(cg, run2, MockLLM()) , "1500 tokens sit under 100000";
    changed = policy.prune(cg, run2, MockLLM());
    assert changed == 2;
    assert states(cg)
        == {"call_dup": "full", "call_a": "full", "call_b": "full", "call_c": "full"} , states(
        cg
    );
    for tr in cg.tool_results() {
        assert tr.visible_content == tr.full_content , f"{tr.call_id} is back at full fidelity";
    }
    assert bigger.stats()["restores"] == 2
        and bigger.stats()["budget_tokens"] == 100000 , bigger.stats();

    # Shrink again: the same renderings come back (memo), nothing is destroyed.
    policy.prune(cg, run, MockLLM());
    assert states(cg)
        == {
            "call_dup": "full",
            "call_a": "head_tail",
            "call_b": "head_tail",
            "call_c": "full"
        };
    assert small.stats()["hard_cleared"] == 0;
}


# 11. Sticky trims and the trim memo.
test "a trimmed rendering is byte-identical across later passes and after a restore-retrim cycle" {
    def scenario(policy: ContextRender) -> tuple[str, str, str] {
        ctx = Context(policy=policy, budget_tokens=1000, persist=False);
        (cg, run) = wired(ctx);
        lines = rows("alpha", 300).split("\n");
        lines[150] = "alpha_row_0150 zebra_marker line";
        lines[200] = "alpha_row_0200 yak_marker line";
        big = "\n".join(lines)[:8000];
        assert "zebra_marker" in big and "yak_marker" in big;
        tr_a = add_result(cg, run, "call_a", big);
        add_result(cg, run, "call_n", blob("newest", 400));
        add_message(cg, run, "user", "find zebra_marker");

        policy.prune(cg, run, MockLLM());
        first = tr_a.visible_content;
        assert tr_a.truncation_state == "head_tail" and TRIM_MARKER in first;
        assert "zebra_marker" in first and "yak_marker" not in first , (
            "the middle keeps the line the focus asks for"
        );

        add_message(cg, run, "user", "find yak_marker");
        policy.prune(cg, run, MockLLM());
        second = tr_a.visible_content;

        ctx.budget_tokens = 100000;
        policy.prune(cg, run, MockLLM());
        assert tr_a.truncation_state == "full" and tr_a.visible_content == big , "restored";

        ctx.budget_tokens = 1000;
        policy.prune(cg, run, MockLLM());
        assert tr_a.truncation_state == "head_tail";
        return (first, second, tr_a.visible_content);
    }

    (first, second, retrimmed) = scenario(ContextRender());
    assert second == first , "sticky: a later pass under a drifted focus keeps the same bytes";
    assert retrimmed == first , "memo: a restore then re-trim replays the first rendering";
    assert "yak_marker" not in retrimmed;

    (ctrl_first, ctrl_second, ctrl_retrimmed) = scenario(
        ContextRender(memo_trim=False)
    );
    assert ctrl_second == ctrl_first , "sticky still holds without the memo";
    assert ctrl_retrimmed != ctrl_first , "without the memo the re-trim re-ranks against the new focus";
    assert "yak_marker" in ctrl_retrimmed and "zebra_marker" not in ctrl_retrimmed;
}


# 12. MMR diversity.
test "the MMR fill prefers a distinct old result over a near-duplicate of one already kept" {
    for (label, lam, want_stubbed) in [
        ("with the diversity penalty", 0.5, "call_p1"),
        ("without it, recency wins", 0.0, "call_q")
    ] {
        policy = ContextRender(
            stub_overflow="clear", grace_turns=1, pin_recent=0, mmr_lambda=lam
        );
        ctx = Context(policy=policy, budget_tokens=790, persist=False);
        (cg, run) = wired(ctx);
        p1 = blob("pdup", 800);
        p2 = p1.replace("pdup_row_0003", "pdup_row_8888");
        assert len(p1) == len(p2) and p1 != p2;
        add_result(cg, run, "call_q", blob("quiet", 800));
        add_result(cg, run, "call_p1", p1);
        add_result(cg, run, "call_p2", p2);
        add_result(cg, run, "call_n", blob("newest", 800));

        sim = lexical_set_cosine(cached_lextoks(p1), cached_lextoks(p2));
        assert sim > 0.9 , f"near-duplicates: {sim}";
        assert lexical_set_cosine(
            cached_lextoks(p1), cached_lextoks(blob("quiet", 800))
        )
            == 0.0;

        policy.prune(cg, run, MockLLM());
        st = states(cg);
        assert st["call_n"] == "full" and st["call_p2"] == "full" , f"{label}: {st}";
        assert st[want_stubbed] == "head_tail" , f"{label}: {st}";
        others = [
            k
            for k in ["call_q", "call_p1"]
            if k != want_stubbed
        ];
        assert st[others[0]] == "full" , f"{label}: {st}";
        assert visible_tokens(cg) <= 790;
    }
}


# 13. Micro stubs.
test "a small result under pressure becomes an identifier-bearing restorable stub; a tiny one is left alone" {
    ident_lines = [
        "order_id=ORD-77812",
        "sku=WIDGET_PRO_2200 qty=3",
        "tracking=1Z999AA10123456784",
        "warehouse=WH_EAST_07 lane=B12",
        "invoice=INV-2026-000913"
    ];
    prose = [
        "the customer asked for a gift receipt and a handwritten note with the parcel please",
        "delivery went smoothly and the neighbour signed for it in the afternoon as agreed",
        "we should remember to follow up next week about the loyalty programme enrolment",
        "nothing else of note happened during the handover and the driver left on time",
        "thanks again for shopping with us and have a pleasant rest of the week and beyond"
    ];
    small = "\n".join(ident_lines + prose);
    small = (small + "\n" + "filler words " * 40)[:600];
    assert len(small) == 600;
    tiny = "order_9=ORD-77899 status=shipped carrier=UPS-GROUND-9 note=leave at the side door";
    assert len(tiny) <= 100 , len(tiny);

    llm = MockLLM(
        outputs=[
            call("echo", {"text": tiny}, call_id="call_t"),
            call("echo", {"text": small}, call_id="call_s"),
            call("fetch", {"key": "beta", "chars": 300}, call_id="call_b"),
            finish("done")
        ]
    );
    def echo(text: str) -> str {
        return text;
    }
    ctx = Context(budget_tokens=150, persist=False);

    def agent(task: str) -> str by llm(tools=[echo, fetch], context=ctx);

    assert agent("look at the orders") == "done";
    assert len(llm.seen) == 4;
    last = sent_messages(llm, 3);
    stub = tool_text(last, "call_s");
    assert stub != small and TRIM_MARKER in stub , f"the 600-char result is stubbed: {stub!r}";
    kept = stub_lines(stub);
    assert kept == ident_lines , f"the stub keeps exactly the identifier-bearing lines, in order: {kept}";
    assert len("\n".join(kept)) <= 150 , "within micro_stub_chars";
    for word in ["customer", "neighbour", "loyalty", "filler"] {
        assert word not in stub , f"prose line dropped: {word}";
    }
    assert tool_text(last, "call_t") == tiny , "a result under micro_stub_min is not stubbed";
    assert tool_text(last, "call_b") == blob("beta", 300) , "the newest is full";

    cg = ctx.graph;
    assert cg is not None;
    tr_s = by_call(cg)["call_s"];
    assert tr_s.truncation_state == "head_tail" and tr_s.full_content == small , "restorable";
    assert by_call(cg)["call_t"].truncation_state == "full";
    s = ctx.stats();
    assert s["stubs"] == 1
        and s["trims"] == 0
        and s["clears"] == 0
        and s["hard_cleared"] == 0 , s;
    assert visible_tokens(cg) <= 150;
}


# 14. Stub overflow.
test "stub_overflow=clear escalates to a named hard clear only when the stubs alone would not fit" {
    tiny_results = [
        f"order_{i}=ORD-7781{i} status=shipped carrier=UPS-GROUND-9 note=leave at the side door"
        for i in range(6)
    ];
    stubbable = [blob(f"r{i}", 600) for i in range(6)];
    for (
        label,
        overflow,
        allow,
        budget,
        contents,
        want_named,
        want_stubbed,
        want_bare
    ) in [
        (
            "clear, tiny results, stubs would overflow",
            "clear",
            False,
            60,
            tiny_results,
            5,
            0,
            0
        ),
        ("clear, tiny results, stubs fit", "clear", False, 300, tiny_results, 0, 0, 0),
        (
            "keep, stub-able results, stubs fit the budget",
            "keep",
            False,
            400,
            stubbable,
            0,
            5,
            0
        ),
        (
            "keep, stubs overflow the budget, hard clear not allowed",
            "keep",
            False,
            240,
            stubbable,
            0,
            5,
            0
        ),
        (
            "keep, stubs overflow the budget, hard clear allowed",
            "keep",
            True,
            240,
            stubbable,
            0,
            0,
            5
        )
    ] {
        policy = ContextRender(stub_overflow=overflow, allow_hard_clear=allow);
        ctx = Context(policy=policy, budget_tokens=budget, persist=False);
        (cg, run) = wired(ctx);
        for i in range(6) {
            add_result(cg, run, f"call_{i}", contents[i]);
        }

        policy.prune(cg, run, MockLLM());
        st = states(cg);
        named: list[str] = [];
        bare: list[str] = [];
        for k in st {
            if st[k] != "hard_cleared" {
                continue;
            }
            text = by_call(cg)[k].visible_content;
            if text == HARD_CLEAR_PLACEHOLDER {
                bare.append(k);
            } else {
                assert text.startswith("[evicted from context: fetch(")
                    and "Re-run this call" in text , (
                    f"{label}: an escalated result names its call: {text}"
                );
                named.append(k);
            }
        }
        stubbed = [
            k
            for k in st
            if st[k] == "head_tail"
        ];
        assert (len(named), len(stubbed), len(bare))
            == (want_named, want_stubbed, want_bare) , (
            f"{label}: named {named}, stubbed {stubbed}, bare {bare}"
        );
        assert st["call_5"] == "full" , f"{label}: the newest is never cleared: {st}";
        assert ctx.stats()["clears"] == want_named + want_bare , f"{label}: {ctx.stats()}";
        assert ctx.stats()["stubs"] == want_stubbed + want_bare , (
            f"{label}: under keep every evictee is stubbed first; the backstop clears the "
            f"oldest stubs only when hard clears are allowed: {ctx.stats()}"
        );
        if want_bare {
            assert visible_tokens(cg) <= budget , f"{label}: the backstop stops at the budget";
        } elif want_stubbed and budget == 240 {
            assert visible_tokens(cg) > budget , (
                f"{label}: without hard clears the budget is a target, nothing is deleted"
            );
        }
    }
}


# 15. Embeddings.
test "with embed_model relevance comes from litellm embeddings; a failure falls back to lexical until the cooldown ends" {
    EMBED_CACHE.clear();
    EMBED_STATE.clear();
    try {
        policy = ContextRender(embed_model="fake-embed");
        ctx = Context(policy=policy, budget_tokens=10000, persist=False);
        (cg, run) = wired(ctx);
        tr_a = add_result(cg, run, "call_a", blob("alpha", 300));
        tr_b = add_result(cg, run, "call_b", blob("beta", 300));
        lexical = ContextRender().candidate_scores(cg, run);
        assert abs(lexical[tr_a.tr_id] - ContextRender().base_score(2, 0.0, 0.0))
            < 1e-12 , ("lexically neither result overlaps the task");

        seen_inputs: list[str] = [];
        def fake_embedding(
            model: str, input: str, timeout: float, **extra: any
        ) -> dict[str, any] {
            seen_inputs.append(input);
            aligned = ("alpha" in input) or ("Do the thing" in input);
            return {"data": [{"embedding": [1.0, 0.0] if aligned else [0.0, 1.0]}]};
        }

        with mock.patch.object(litellm, "embedding", side_effect=fake_embedding) as m {
            scores = policy.candidate_scores(cg, run);
        }
        assert m.call_count == 3 , f"query text once (anchor and focus coincide), then each result: {seen_inputs}";
        assert m.call_args.kwargs["model"] == "fake-embed";
        assert m.call_args.kwargs["timeout"] == 60.0;
        assert blob("beta", 300) in seen_inputs and "Do the thing." in seen_inputs;
        assert abs(scores[tr_a.tr_id] - policy.base_score(2, 0.0, 1.0)) < 1e-9 , (
            f"alpha's vector is the query's: relevance 1: {scores}"
        );
        assert abs(scores[tr_b.tr_id] - policy.base_score(0, 0.0, 0.0)) < 1e-9 , (
            f"beta is orthogonal: relevance 0: {scores}"
        );

        with mock.patch.object(
            litellm, "embedding", side_effect=fake_embedding
        ) as again {
            policy.candidate_scores(cg, run);
        }
        assert again.call_count == 0 , "embeddings are cached by content";

        EMBED_CACHE.clear();
        with mock.patch.object(
            litellm, "embedding", side_effect=RuntimeError("no key")
        ) as broken {
            fallback = policy.candidate_scores(cg, run);
            assert broken.call_count == 2 , f"one retry, then give up: {broken.call_count}";
            assert EMBED_STATE.get("fake-embed", 0.0) > time.time() , (
                "the model is rested for a while, not for the rest of the process"
            );
            fallback_again = policy.candidate_scores(cg, run);
        }
        assert broken.call_count == 2 , "a pass inside the cooldown does not call the provider";
        assert fallback == lexical and fallback_again == lexical , (
            f"the whole pass is lexical, never a mix: {fallback} vs {lexical}"
        );
        assert len(EMBED_CACHE) == 0 , "a failure is not cached as an answer";

        # Once the cooldown is over the provider is asked again and embeddings resume.
        EMBED_STATE["fake-embed"] = time.time() - 1.0;
        with mock.patch.object(
            litellm, "embedding", side_effect=fake_embedding
        ) as healed {
            resumed = policy.candidate_scores(cg, run);
        }
        assert healed.call_count == 3 and resumed == scores , f"{resumed} vs {scores}";
        assert "fake-embed" not in EMBED_STATE;

        # An empty result is not sent to the provider and does not fail the pass.
        tr_e = add_result(cg, run, "call_e", "");
        with mock.patch.object(
            litellm, "embedding", side_effect=fake_embedding
        ) as with_empty {
            mixed = policy.candidate_scores(cg, run);
        }
        assert with_empty.call_count == 0 , "nothing new to embed: the empty text is skipped";
        assert "fake-embed" not in EMBED_STATE , "embeddings stay on";
        assert abs(mixed[tr_a.tr_id] - policy.base_score(4, 0.0, 1.0)) < 1e-9 , (
            f"alpha is still scored by its embedding: {mixed}"
        );
        assert abs(mixed[tr_e.tr_id] - policy.base_score(0, 0.0, 0.0)) < 1e-9;

        # The key and base are passed only when the policy sets them.
        assert "api_key" not in healed.call_args.kwargs
            and "api_base" not in healed.call_args.kwargs;
        EMBED_CACHE.clear();
        keyed = ContextRender(
            embed_model="fake-embed",
            embed_api_key="sk-embed",
            embed_api_base="http://proxy/v1"
        );
        with mock.patch.object(
            litellm, "embedding", side_effect=fake_embedding
        ) as routed {
            keyed.candidate_scores(cg, run);
        }
        assert routed.call_args.kwargs["api_key"] == "sk-embed";
        assert routed.call_args.kwargs["api_base"] == "http://proxy/v1";
    } finally {
        EMBED_CACHE.clear();
        EMBED_STATE.clear();
    }
}


# 16. Churn controls.
test "the minimum-gain gate leaves the graph untouched, and hysteresis favours what is already visible" {
    for (label, floor, want_state, want_prunes) in [
        ("gate on, removal too small", 1000000, "full", 0),
        ("gate off", 0, "head_tail", 1)
    ] {
        policy = ContextRender(evict_min_tokens=floor);
        ctx = Context(policy=policy, budget_tokens=300, persist=False);
        (cg, run) = wired(ctx);
        tr_a = add_result(cg, run, "call_a", blob("alpha", 800));
        add_result(cg, run, "call_n", blob("newest", 800));
        changed = policy.prune(cg, run, MockLLM());
        assert tr_a.truncation_state == want_state , f"{label}: {states(cg)}";
        if want_state == "full" {
            assert tr_a.visible_content == blob("alpha", 800) , f"{label}: untouched";
        }
        assert ctx.stats()["prunes"] == want_prunes , f"{label}: {ctx.stats()}";
        assert changed == (0 if want_state == "full" else 1) , f"{label}: {changed}";
    }

    ratio = ContextRender(evict_min_gain_ratio=100.0);
    ctx = Context(policy=ratio, budget_tokens=300, persist=False);
    (cg, run) = wired(ctx);
    tr_a = add_result(cg, run, "call_a", blob("alpha", 800));
    add_result(cg, run, "call_n", blob("newest", 800));
    assert ratio.prune(cg, run, MockLLM()) == 0 and tr_a.truncation_state == "full" , "ratio gate";

    # A restore is never gated: the gate only skips passes that would purely remove.
    gated = ContextRender(evict_min_tokens=1000000);
    ctx = Context(policy=gated, budget_tokens=300, persist=False);
    (cg, run) = wired(ctx);
    tr_a = add_result(cg, run, "call_a", blob("alpha", 800));
    add_result(cg, run, "call_n", blob("newest", 800));
    ContextRender().prune(cg, run, MockLLM());
    assert tr_a.truncation_state == "head_tail";
    ctx.budget_tokens = 100000;
    assert gated.prune(cg, run, MockLLM()) == 1 and tr_a.truncation_state == "full";

    # Hysteresis: a margin for results that are currently full.
    ctx = Context(policy=ContextRender(), budget_tokens=10000, persist=False);
    (cg, run) = wired(ctx);
    tr_a = add_result(cg, run, "call_a", blob("alpha", 800));
    tr_b = add_result(cg, run, "call_b", blob("beta", 800));
    tr_a.truncation_state = "head_tail";
    plain = ContextRender().candidate_scores(cg, run);
    sticky = ContextRender(select_hysteresis=0.5).candidate_scores(cg, run);
    assert abs(sticky[tr_a.tr_id] - plain[tr_a.tr_id]) < 1e-12 , "a stubbed result gets no margin";
    assert abs(sticky[tr_b.tr_id] - plain[tr_b.tr_id] - 0.5) < 1e-12 , "a full one gets the margin";
}


# 17. A persisted Context across separate calls.
test "a second call on a persisted Context budgets, trims and anchors on its own transcript only" {
    policy = ContextRender();
    ctx = Context(policy=policy, persist=True, budget_tokens=100000);
    llm_one = MockLLM(
        outputs=[
            call("fetch", {"key": "alpha", "chars": 8000}, call_id="call_a"),
            call("fetch", {"key": "beta", "chars": 8000}, call_id="call_b"),
            finish("one")
        ]
    );

    def agent_one(task: str) -> str by llm_one(tools=[fetch], context=ctx);

    assert agent_one("collect alpha and beta") == "one";
    cg = ctx.graph;
    assert cg is not None;
    assert states(cg) == {"call_a": "full", "call_b": "full"};

    # The second call carries no conversation, so alpha and beta are on the graph but not in
    # its transcript: 4000 invisible tokens must not fire the trigger nor be spent or trimmed
    # (at 150 they would be smart-trimmed at once; gamma + delta = 225 still trip the pass).
    ctx.budget_tokens = 150;
    llm_two = MockLLM(
        outputs=[
            call("fetch", {"key": "gamma", "chars": 600}, call_id="call_c"),
            call("fetch", {"key": "delta", "chars": 300}, call_id="call_d"),
            finish("two")
        ]
    );

    def agent_two(task: str) -> str by llm_two(tools=[fetch], context=ctx);

    assert agent_two("collect gamma and delta") == "two";
    assert ctx.graph is cg;
    assert len(llm_two.seen) == 3;
    for n in range(3) {
        sent = sent_messages(llm_two, n);
        assert not has_tool_message(sent, "call_a")
            and not has_tool_message(sent, "call_b") , (
            f"call {n}: the first call's results never reach the model"
        );
    }
    last = sent_messages(llm_two, 2);
    assert TRIM_MARKER in tool_text(last, "call_c") , "gamma (150 tokens) is stubbed under the 150 budget";
    assert tool_text(last, "call_d") == blob("delta", 300) , "delta, the newest, is full";
    assert states(cg)
        == {
            "call_a": "full",
            "call_b": "full",
            "call_c": "head_tail",
            "call_d": "full"
        } , states(cg);
    assert by_call(cg)["call_a"].visible_content == blob("alpha", 8000) , "alpha was never rewritten";
    s = ctx.stats();
    assert s["budget_tokens"] == 150
        and s["stubs"] == 1
        and s["trims"] == 0
        and s["clears"] == 0 , s;
    assert s["full"] == 3 and s["trimmed"] == 1 and s["hard_cleared"] == 0 , s;
    assert s["visible_tokens"] > 4000 , "stats() still reports the whole graph";

    # A third transcript on the same graph, built by hand, shows the presence rule directly:
    # its anchor is its own task and its visible tool tokens are its own results.
    third = Context(policy=policy, budget_tokens=100, persist=True);
    third.graph = cg;
    (_, run) = wired(third, task="third task: find epsilon");
    add_result(cg, run, "call_e", blob("epsilon", 400));
    present = presence_of(run);
    assert focus_texts(cg, present)["anchor"] == "third task: find epsilon" , focus_texts(
        cg, present
    );
    assert visible_tool_tokens(cg, present) == 100 , visible_tool_tokens(cg, present);
    assert not policy.should_prune(cg, run, MockLLM()) , "100 tokens do not exceed 100";
    unrestricted = presence_of(object());
    assert unrestricted.unrestricted and visible_tool_tokens(cg, unrestricted) > 4000 , (
        "a runtime without a message list is not restricted"
    );
}


# 18. Typed compaction through the threshold path.
test "compaction fires through the threshold, folds older turns into typed slots and keeps the prefix" {
    slots = CompactionSlots(
        files_read=["config/prod.yaml"], key_observations=[VALUE_LINE]
    );
    llm = MockLLM(
        ctx_window=1000,
        outputs=[
            call(
                "fetch",
                {"key": "alpha", "chars": 200},
                call_id="call_a",
                usage={"prompt_tokens": 100}
            ),
            call(
                "fetch",
                {"key": "beta", "chars": 200},
                call_id="call_b",
                usage={"prompt_tokens": 850}
            ),
            slots,
            finish("ok")
        ]
    );
    ctx = Context(budget_tokens=100000, persist=False);

    def agent(task: str) -> str by llm(
        tools=[fetch], context=ctx, threshold_ratio=0.8, keep_recent_iterations=1
    );

    assert agent("find the keepalive") == "ok";
    assert len(llm.seen) == 4 , f"two tool rounds, the extractor, the finish: {len(
        llm.seen
    )}";
    assert llm.exhausted();

    assert roles(llm, 2) == ["system", "user"] , "the extractor call carries the instructions and the JSON";
    body = str(sent_messages(llm, 2)[1]["content"]);
    assert body.startswith("{") and '"items"' in body;
    assert '"role": "tool:fetch"' in body and json.dumps(blob("alpha", 200)) in body , body[:300];
    assert '"turn": 3' in body and '"turn": 4' in body and "alpha" in body;
    assert "called: fetch(" in body , "a tool-call turn with no prose is summarised as its call";
    assert "beta" not in body , "the round being kept is not folded";
    assert body.count("previous_summary") == 1 , "the previous summary field is sent once";
    assert llm.sent("response_format")[2] is not None , "a typed CompactionSlots reply is requested";

    first = sent_messages(llm, 0);
    after = sent_messages(llm, 3);
    assert [m["role"] for m in after]
        == ["system", "user", "user", "assistant", "tool"] , (
        f"system, task, summary, then the kept round: {[m['role'] for m in after]}"
    );
    assert after[0] == first[0] and after[1] == first[1] , "the system and task messages are still first";
    summary = str(after[2]["content"]);
    assert summary.startswith(
        COMPACTION_TAG + "\n[Typed compaction summary]"
    ) , summary;
    assert VALUE_LINE in summary and "Files read: config/prod.yaml" in summary , "values are preserved verbatim";
    assert not has_tool_message(after, "call_a") , "the folded tool result is gone";
    assert tool_text(after, "call_b") == blob("beta", 200);
    assert [tc["id"] for tc in after[3]["tool_calls"]] == ["call_b"];

    cg = ctx.graph;
    assert cg is not None;
    s = ctx.stats();
    assert s["compactions"] == 1 and s["summaries"] == 1 , s;
    assert s["hidden_messages"] == 1 and s["hidden_tool_results"] == 1 , s;
    summ = cg.summaries()[0];
    assert (summ.covers_turn_start, summ.covers_turn_end) == (3, 4) , (
        f"the call at turn 3 and its result at turn 4: {(
            summ.covers_turn_start,
            summ.covers_turn_end
        )}"
    );
    assert summ.dropped_message_count == 1 and summ.dropped_tool_result_count == 1;
    assert [
        m.role
        for m in cg.messages()
        if not m.visible
    ]
        == ["assistant"];
    assert cg.messages()[0].visible and cg.messages()[1].visible , "system and first user are never hidden";
}


# 19. The emergency path.
test "a ContextWindowExceededError routes the emergency compaction to the policy and retries" {
    llm = MockLLM(
        outputs=[
            call("fetch", {"key": "alpha", "chars": 200}, call_id="call_a"),
            fail(
                litellm.exceptions.ContextWindowExceededError(
                    message="ctx exceeded", model="mockllm", llm_provider="openai"
                )
            ),
            CompactionSlots(key_observations=[VALUE_LINE]),
            finish("ok")
        ]
    );
    ctx = Context(budget_tokens=100000, persist=False);

    def agent(task: str) -> str by llm(
        tools=[fetch], context=ctx, keep_recent_iterations=0
    );

    assert agent("find the keepalive") == "ok";
    assert len(llm.seen) == 4 , f"round one, the failed call, the extractor, the retry: {len(
        llm.seen
    )}";
    assert roles(llm, 2) == ["system", "user"];
    assert json.dumps(blob("alpha", 200)) in str(sent_messages(llm, 2)[1]["content"]);
    retry = sent_messages(llm, 3);
    assert [m["role"] for m in retry] == ["system", "user", "user"] , [
        m["role"] for m in retry
    ];
    assert str(retry[2]["content"]).startswith(COMPACTION_TAG)
        and VALUE_LINE in str(retry[2]["content"]);
    assert not has_tool_message(retry, "call_a");
    assert ctx.stats()["compactions"] == 1 and ctx.stats()["hidden_tool_results"] == 1;
}


# 20. The hook wins.
test "an on_compaction hook still wins over the policy when both are set" {
    llm = MockLLM(
        ctx_window=1000,
        outputs=[
            call(
                "fetch",
                {"key": "alpha", "chars": 200},
                call_id="call_a",
                usage={"prompt_tokens": 100}
            ),
            call(
                "fetch",
                {"key": "beta", "chars": 200},
                call_id="call_b",
                usage={"prompt_tokens": 850}
            ),
            finish("ok")
        ]
    );
    hook_calls = [0];
    def hook(msgs: list[any], keep_recent: int) -> list[any] {
        hook_calls[0] += 1;
        return msgs[:2];
    }
    ctx = Context(budget_tokens=100000, persist=False);

    def agent(task: str) -> str by llm(
        tools=[fetch], context=ctx, threshold_ratio=0.8, on_compaction=hook
    );

    assert agent("find the keepalive") == "ok";
    assert hook_calls[0] == 1 , "the hook compacted";
    assert len(llm.seen) == 3 , f"no extractor call was made: {len(llm.seen)}";
    assert llm.exhausted();
    s = ctx.stats();
    assert s["compactions"] == 0
        and s["summaries"] == 0
        and s["hidden_messages"] == 0 , s;
    assert [m["role"] for m in sent_messages(llm, 2)][:2] == ["system", "user"];
    assert COMPACTION_TAG not in str(sent_messages(llm, 2));
}


# 21. A compaction that cannot finish changes nothing.
test "a failed extraction or an empty span raises and leaves the graph as it was" {
    policy = ContextRender();
    ctx = Context(policy=policy, budget_tokens=100000, persist=False);
    (cg, run) = wired(ctx, task="find the keepalive");
    add_result(cg, run, "call_a", blob("alpha", 200));
    add_result(cg, run, "call_b", blob("beta", 200));
    add_result(cg, run, "call_c", blob("gamma", 200));

    good = MockLLM(outputs=[CompactionSlots(key_observations=[VALUE_LINE])]);
    assert policy.compact(cg, run, good, 2) , "the first round folds";
    before = ctx.stats();
    assert before["compactions"] == 1 and before["hidden_tool_results"] == 1 , before;
    kept = [
        s.summary_text
        for s in cg.summaries()
        if s.visible
    ];
    assert len(kept) == 1 and VALUE_LINE in kept[0];

    broken = MockLLM(outputs=[fail(RuntimeError("extractor down"))]);
    raised = "";
    try {
        policy.compact(cg, run, broken, 1);
    } except RuntimeError as e {
        raised = str(e);
    }
    assert raised == "extractor down" , f"the error reaches the caller: {raised!r}";
    assert broken.exhausted();
    assert ctx.stats() == before , f"nothing was hidden or counted: {ctx.stats()}";
    assert [
        s.summary_text
        for s in cg.summaries()
        if s.visible
    ]
        == kept , ("the earlier summary is still the visible one");
    assert VALUE_LINE in str(policy.materialize(cg, run));

    # Fewer assistant rounds than keep_recent: nothing to fold, and no model call.
    idle = MockLLM(outputs=[]);
    declined = False;
    try {
        policy.compact(cg, run, idle, 5);
    } except CompactionNotEffectiveError as e {
        declined = "nothing to fold" in str(e);
    }
    assert declined and len(idle.seen) == 0;
    assert ctx.stats() == before;
}


# 22. What the extractor reads.
test "the extractor reads full results while they fit its window, and the trimmed view past that" {
    def folded_body(window: int) -> tuple[str, ContextToolResult] {
        policy = ContextRender();
        ctx = Context(policy=policy, budget_tokens=100000, persist=False);
        (cg, run) = wired(ctx, task="find the keepalive");
        tr = add_result(cg, run, "call_a", blob("alpha", 8000));
        add_result(cg, run, "call_b", blob("beta", 200));
        apply_eviction_decision(
            EvictionDecision(
                tool_result=tr, kind="soft_trim", head_chars=100, tail_chars=100
            )
        );
        llm = MockLLM(ctx_window=window, outputs=[CompactionSlots()]);
        assert policy.compact(cg, run, llm, 1);
        return (str(sent_messages(llm, 0)[1]["content"]), tr);
    }

    # 8000 chars are 2000 estimated tokens: half of a 100000 window holds them.
    (roomy, tr_full) = folded_body(100000);
    assert json.dumps(blob("alpha", 8000)) in roomy , "the full text is read when it fits";

    # Half of a 1000 window does not, so the extractor gets what the model was seeing.
    (tight, tr_view) = folded_body(1000);
    assert tr_view.visible_content != tr_view.full_content
        and len(tr_view.visible_content) < 300;
    assert json.dumps(tr_view.visible_content) in tight , tight[:300];
    assert json.dumps(blob("alpha", 8000)) not in tight and len(tight) < 2000 , len(
        tight
    );
    assert tr_full.full_content == blob("alpha", 8000) , "the stored result is never shortened";
}


# 23. Results with no usable line breaks.
test "a result made of one long line, or led by one, is still trimmed and stubbed" {
    one_line = "{"
        + ", ".join([f"'key_{i}': 'value_{i * 7}'" for i in range(60)])
        + "}";
    assert "\n" not in one_line and 1000 < len(one_line) < 3000;
    policy = ContextRender();
    ctx = Context(policy=policy, budget_tokens=200, persist=False);
    (cg, run) = wired(ctx);
    tr = add_result(cg, run, "call_a", one_line);
    add_result(cg, run, "call_n", blob("newest", 400));
    policy.prune(cg, run, MockLLM());
    stub = tr.visible_content;
    assert tr.truncation_state == "head_tail"
        and len(stub) <= policy.micro_stub_chars , len(stub);
    assert stub.split("\n")[1] == TRIM_MARKER , stub;
    assert stub.startswith(one_line[:40]) and stub.endswith(one_line[-40:]) , stub;
    policy.prune(cg, run, MockLLM());
    assert tr.visible_content == stub , "the stub is byte-stable across passes";

    wide = ("x" * 50000) + "\n" + rows("tail", 200);
    roomy = Context(policy=policy, budget_tokens=2000, persist=False);
    (cg2, run2) = wired(roomy);
    tr_wide = add_result(cg2, run2, "call_w", wide);
    add_result(cg2, run2, "call_n", blob("newest", 400));
    policy.prune(cg2, run2, MockLLM());
    trimmed = tr_wide.visible_content;
    assert tr_wide.truncation_state == "head_tail";
    assert len(trimmed) <= 2 * policy.trim_chars , f"one 50000-char line cannot ride along: {len(
        trimmed
    )}";
    assert trimmed.startswith("xxxx")
        and trimmed.endswith(wide[-40:])
        and TRIM_MARKER in trimmed;
    assert roomy.stats()["hard_cleared"] == 0;
}


# 24. The backstop under the default settings.
test "by default pressure turns trimmed results into restorable stubs and deletes nothing" {
    policy = ContextRender();
    ctx = Context(policy=policy, budget_tokens=1500, persist=False);
    (cg, run) = wired(ctx);
    for i in range(9) {
        add_result(cg, run, f"call_{i}", blob(f"r{i}", 4000));
    }
    policy.prune(cg, run, MockLLM());
    st = states(cg);
    assert "hard_cleared" not in st.values() and st["call_8"] == "full" , st;
    assert visible_tool_tokens(cg, presence_of(run)) <= 1500 , (
        "eight 750-token trims would not fit, so the oldest were shrunk to stubs"
    );
    s = ctx.stats();
    assert s["clears"] == 0 and s["hard_cleared"] == 0 and s["stubs"] > 0 , s;
    assert len(by_call(cg)["call_0"].visible_content) < 300 , (
        "a stub is its 150 chars of lines plus the elision markers"
    );

    ctx.budget_tokens = 100000;
    policy.prune(cg, run, MockLLM());
    assert set(states(cg).values()) == {"full"} , "every result comes back when there is room";
    assert by_call(cg)["call_0"].visible_content == blob("r0", 4000);
}


# 25. Citations on a persisted graph.
test "a citation credits a result of the current call, not an older call's copy of the identifier" {
    policy = ContextRender();
    ctx = Context(policy=policy, budget_tokens=100000, persist=True);
    (cg, run_one) = wired(ctx, task="first task");
    add_result(
        cg, run_one, "call_old", "config: prod.yaml sets keepalive_seconds to 47"
    );
    (same, run_two) = wired(ctx, task="second task");
    assert same is cg;
    add_result(
        cg, run_two, "call_live", "config: prod.yaml sets keepalive_seconds to 47 again"
    );
    add_result(cg, run_two, "call_cite", "ok", args={"path": "prod.yaml"});

    scan_citations(cg, 20000, presence_of(run_two));
    assert (cg.citation_counts.get("call_live") or 0.0) > 0.0 , cg.citation_counts;
    assert not cg.citation_counts.get("call_old") , (
        f"the older call's result is not in this prompt: {cg.citation_counts}"
    );
}


# 26. A turn with several tool calls.
test "every result of a multi-call turn reaches the model in full before any is trimmed" {
    def run_batch(policy: ContextRender) -> tuple[list[dict[str, any]], Context] {
        llm = MockLLM(
            outputs=[
                calls(
                    [
                        ("fetch", {"key": f"k{i}", "chars": 2000}, f"call_{i}")
                        for i in range(4)
                    ]
                ),
                finish("ok")
            ]
        );
        ctx = Context(policy=policy, budget_tokens=1500, persist=False);

        def agent(task: str) -> str by llm(tools=[fetch], context=ctx);

        assert agent("collect four things") == "ok";
        return (sent_messages(llm, 1), ctx);
    }

    (sent, ctx) = run_batch(ContextRender());
    for i in range(4) {
        assert tool_text(sent, f"call_{i}") == blob(f"k{i}", 2000) , (
            f"call_{i} was shortened before the model read it"
        );
    }
    s = ctx.stats();
    assert s["trims"] == 0 and s["stubs"] == 0 and s["clears"] == 0 , s;

    # With the guard off the old behaviour is back: only the newest is exempt.
    (unguarded, _) = run_batch(ContextRender(protect_current_turn=False));
    assert tool_text(unguarded, "call_3") == blob("k3", 2000);
    assert any(
        tool_text(unguarded, f"call_{i}") != blob(f"k{i}", 2000) for i in range(3)
    );
}


# 27. Where the budget ratio comes from.
test "the budget ratio is the Context's when set, else the configured one" {
    policy = ContextRender();
    model = MockLLM(ctx_window=1000);
    run = mk_run(tools=[fetch]);
    assert policy._resolve_budget(Context(policy=policy), run, model) == 500 , (
        "with nothing set, half the window"
    );
    assert policy._resolve_budget(Context(policy=policy, budget_ratio=0.2), run, model)
        == 200;

    configured = FakeConfig(context={"budget_tokens": 0, "budget_ratio": 0.3});
    with mock.patch.object(
        render_module, "get_byllm_config", return_value=configured
    ) {
        assert policy._resolve_budget(Context(policy=policy), run, model) == 300 , (
            "[plugins.byllm.context] budget_ratio applies when the Context sets none"
        );
        assert policy._resolve_budget(
            Context(policy=policy, budget_ratio=0.2), run, model
        )
            == 200 , "the Context wins";
        configured.context["budget_tokens"] = 250;
        assert policy._resolve_budget(
            Context(policy=policy, budget_ratio=0.2), run, model
        )
            == 250 , "a configured token budget beats any ratio";
        assert policy._resolve_budget(
            Context(policy=policy, budget_tokens=70), run, model
        )
            == 70;
    }
}


# 28. A hook compacts the real history.
test "an on_compaction hook reads and returns the canonical history, never the trimmed view" {
    policy = ContextRender();
    ctx = Context(policy=policy, budget_tokens=100000, persist=False);
    (cg, run) = wired(ctx, task="find the keepalive");
    tr_a = add_result(cg, run, "call_a", blob("alpha", 8000));
    tr_b = add_result(cg, run, "call_b", blob("beta", 8000));
    apply_eviction_decision(
        EvictionDecision(
            tool_result=tr_b, kind="soft_trim", head_chars=100, tail_chars=100
        )
    );
    assert tool_text(run.get_msg_list(), "call_b") == tr_b.visible_content , "the view is trimmed";

    given: list[any] = [];
    def hook(msgs: list[any], keep_recent: int) -> list[any] {
        given.extend(msgs);
        return msgs[:2] + msgs[-2:];
    }
    run.call_params["on_compaction"] = hook;
    MockLLM()._compact_messages(run, 1, None);

    assert tool_text(given, "call_b") == blob("beta", 8000) , "the hook is handed the full text";
    assert len(run.messages) == 4
        and tool_text(run.messages, "call_b") == blob("beta", 8000) , (
        "what the hook returns is the history, untrimmed"
    );
    present = presence_of(run);
    assert focus_texts(cg, present)["anchor"] == "find the keepalive" , (
        "the kept messages still map to their graph nodes"
    );
    assert present.has_result(tr_b) and not present.has_result(tr_a);
    assert tool_text(run.get_msg_list(), "call_b") == tr_b.visible_content , "the view is still the view";
    tr_b.visible_content = tr_b.full_content;
    tr_b.truncation_state = "full";
    assert tool_text(run.get_msg_list(), "call_b") == blob("beta", 8000) , "a restore reaches the model";
}
