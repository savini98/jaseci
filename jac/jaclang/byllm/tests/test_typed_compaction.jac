"""Typed-slot compaction at the unit level: rendering, the extractor call as the
model sees it, coercion of each reply shape, error propagation, and usage bookkeeping.

The end-to-end path through the ReAct loop (compaction firing under a wired
Context) is covered by test_context_render.jac, not here.
"""

import json;
import from jaclang.byllm.lib { MockLLM, MockRawResponse }
import from jaclang.byllm.typed_compaction {
    COMPACTION_EXTRACT_INSTRUCTIONS,
    CompactionSlots,
    coerce_compaction_slots,
    extract_compaction_slots,
    render_compaction_slots
}
import from support { fail, mk_run, roles, sent_messages }


glob VALUE_LINE = "keepalive_seconds = 47 (config/prod.yaml)",
     ITEMS: list[dict[str, any]] = [
         {"turn": 1, "role": "user", "content": "Find the keepalive setting."},
         {
             "turn": 2,
             "role": "tool:read_file",
             "content": "server:\n  keepalive_seconds: 47\n  port: 8080"
         },
         {"turn": 3, "role": "assistant", "content": "The keepalive is 47 seconds."}
     ];


test "render_compaction_slots lays out headers, indents entries, omits empty slots" {
    full = CompactionSlots(
        files_read=["config/prod.yaml", "services/api.py"],
        key_observations=[VALUE_LINE, "PORT = 8080 (config/prod.yaml)"],
        commands_run=["grep -n keepalive config/prod.yaml"],
        decisions=["Keep the 47 second keepalive."],
        unresolved_questions=["Is 8080 also used in staging?"]
    );
    full_text = (
        "[Typed compaction summary]\n"
        "Files read: config/prod.yaml, services/api.py\n"
        "Key observations:\n"
        f"  - {VALUE_LINE}\n"
        "  - PORT = 8080 (config/prod.yaml)\n"
        "Commands run:\n"
        "  - grep -n keepalive config/prod.yaml\n"
        "Decisions:\n"
        "  - Keep the 47 second keepalive.\n"
        "Unresolved:\n"
        "  - Is 8080 also used in staging?"
    );
    for (label, slots, want) in [
        ("every slot filled", full, full_text),
        ("no slots", CompactionSlots(), "[Typed compaction summary]"),
        (
            "one slot",
            CompactionSlots(key_observations=[VALUE_LINE]),
            f"[Typed compaction summary]\nKey observations:\n  - {VALUE_LINE}"
        ),
        (
            "files only, joined on one line",
            CompactionSlots(files_read=["a.py", "b.py"]),
            "[Typed compaction summary]\nFiles read: a.py, b.py"
        )
    ] {
        got = render_compaction_slots(slots);
        assert got == want , f"{label}: got {got!r}";
    }
    rendered = render_compaction_slots(full);
    for absent in ["Unresolved:", "Decisions:", "Commands run:", "Files read:"] {
        assert absent in rendered , f"{absent} must be in the full rendering";
        assert absent
            not in render_compaction_slots(
                CompactionSlots(key_observations=[VALUE_LINE])
            ) , f"{absent} must be omitted when its slot is empty";
    }
}


test "extract_compaction_slots sends the instructions and the items JSON, and returns the typed reply" {
    previous = "[Typed compaction summary]\nKey observations:\n  - PORT = 8080 (a.py)";
    llm = MockLLM(outputs=[CompactionSlots(key_observations=[VALUE_LINE])]);

    slots = extract_compaction_slots(llm, previous, ITEMS);

    assert isinstance(slots, CompactionSlots) , f"got {type(slots).__name__}";
    assert slots.key_observations == [VALUE_LINE] , f"got {slots.key_observations}";
    assert slots.files_read == [] and slots.commands_run == [];
    assert slots.decisions == [] and slots.unresolved_questions == [];

    assert len(llm.seen) == 1 , f"one model call, got {len(llm.seen)}";
    assert llm.exhausted() , "the queued reply must be consumed";
    assert roles(llm) == ["system", "user"] , f"got {roles(llm)}";
    msgs = sent_messages(llm);
    system = msgs[0]["content"];
    assert system == COMPACTION_EXTRACT_INSTRUCTIONS , f"system got {system!r}";
    assert "VERBATIM" in system and "previous_summary" in system;
    body = msgs[1]["content"];
    assert isinstance(body, str) , f"user content got {type(body).__name__}";
    # The items go out as JSON, so the source content arrives JSON-encoded.
    assert json.dumps(ITEMS[1]["content"]) in body , f"items JSON not in {body!r}";
    assert '"role": "tool:read_file"' in body and '"turn": 2' in body;
    assert "PORT = 8080 (a.py)" in body , "the previous summary must go out too";
    assert body.startswith("{") , "the user message is the JSON document";

    # The typed reply is requested as a schema naming the five slots.
    rf = llm.sent("response_format")[0];
    assert isinstance(rf, dict) , "a typed call must request a schema";
    # An object return type is sent as its own schema, unwrapped (schema.impl.jac _wrap_to_object).
    wrapper = rf["json_schema"]["schema"];
    assert set(wrapper["properties"].keys())
        == {
            "files_read",
            "key_observations",
            "commands_run",
            "decisions",
            "unresolved_questions"
        } , f"got {list(wrapper['properties'].keys())}";
    assert llm.sent("tools")[0] in (None, []) , "the extractor offers no tools";
}


test "every reply shape the extractor can see becomes CompactionSlots with the value intact" {
    as_json = json.dumps({"schema_object_wrapper": {"key_observations": [VALUE_LINE]}});
    for (label, reply) in [
        ("a CompactionSlots value", CompactionSlots(key_observations=[VALUE_LINE])),
        ("a dict, parsed upstream into the object", {"key_observations": [VALUE_LINE]}),
        ("raw model JSON", MockRawResponse(content=as_json))
    ] {
        llm = MockLLM(outputs=[reply]);
        slots = extract_compaction_slots(llm, "", ITEMS);
        assert isinstance(slots, CompactionSlots) , f"{label}: got {type(
            slots
        ).__name__}";
        assert slots.key_observations == [VALUE_LINE] , f"{label}: got {slots.key_observations}";
        assert slots.files_read == [] , f"{label}: defaults must hold";
        assert llm.exhausted() , label;
    }

    # The coercion itself handles the shapes a typed path could hand back.
    for (label, value, want_obs, want_files) in [
        (
            "object passthrough",
            CompactionSlots(key_observations=[VALUE_LINE]),
            [VALUE_LINE],
            []
        ),
        ("plain dict", {"key_observations": [VALUE_LINE]}, [VALUE_LINE], []),
        (
            "wrapped dict",
            {"schema_object_wrapper": {"key_observations": [VALUE_LINE]}},
            [VALUE_LINE],
            []
        ),
        ("JSON text", as_json, [VALUE_LINE], []),
        (
            "a lone string widens to a one-entry list",
            {"key_observations": VALUE_LINE, "files_read": "a.py"},
            [VALUE_LINE],
            ["a.py"]
        ),
        ("null slot", {"key_observations": None}, [], [])
    ] {
        got = coerce_compaction_slots(value);
        assert got.key_observations == want_obs , f"{label}: got {got.key_observations}";
        assert got.files_read == want_files , f"{label}: got {got.files_read}";
    }
    for (label, bad) in [("a number", 7), ("a list", [1, 2])] {
        raised = False;
        try {
            coerce_compaction_slots(bad);
        } except TypeError {
            raised = True;
        }
        assert raised , f"{label}: must be rejected";
    }
}


test "a failed extraction raises to the caller, which decides what to hide" {
    llm = MockLLM(outputs=[fail(RuntimeError("boom"))]);
    raised = "";
    try {
        extract_compaction_slots(llm, "", ITEMS);
    } except RuntimeError as e {
        raised = str(e);
    }
    assert raised == "boom" , f"got {raised!r}";
    assert llm.exhausted();

    # A healthy call renders the slots.
    ok = render_compaction_slots(
        extract_compaction_slots(
            MockLLM(outputs=[CompactionSlots(key_observations=[VALUE_LINE])]), "", ITEMS
        )
    );
    assert ok == f"[Typed compaction summary]\nKey observations:\n  - {VALUE_LINE}" , (
        f"got {ok!r}"
    );
}


test "the nested call's usage lands on the parent run's history" {
    usage = {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60};
    parent = mk_run();
    assert parent._usage_history == [];

    llm = MockLLM(outputs=[(CompactionSlots(key_observations=[VALUE_LINE]), usage)]);
    slots = extract_compaction_slots(llm, "", ITEMS, parent_mt_run=parent);

    assert slots.key_observations == [VALUE_LINE];
    assert len(parent._usage_history) == 1 , f"got {parent._usage_history}";
    record = parent._usage_history[0];
    assert record["prompt_tokens"] == 50 and record["total_tokens"] == 60 , f"got {record}";
    assert record["call_kind"] == "compaction" , f"got {record}";

    # A second extraction appends; the first record stays.
    llm2 = MockLLM(outputs=[(CompactionSlots(), {"prompt_tokens": 5})]);
    extract_compaction_slots(llm2, "", ITEMS, parent_mt_run=parent);
    assert len(parent._usage_history) == 2 , f"got {parent._usage_history}";
    assert parent._usage_history[1]["prompt_tokens"] == 5;

    # Without a parent the call still works.
    no_parent = extract_compaction_slots(
        MockLLM(outputs=[CompactionSlots()]), "", ITEMS
    );
    assert no_parent.key_observations == [];
}
