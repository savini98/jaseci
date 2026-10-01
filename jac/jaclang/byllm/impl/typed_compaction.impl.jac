impl render_compaction_slots(slots: CompactionSlots) -> str {
    parts: list[str] = ["[Typed compaction summary]"];
    if slots.files_read {
        parts.append("Files read: " + ", ".join(slots.files_read));
    }
    if slots.key_observations {
        parts.append("Key observations:");
        for obs in slots.key_observations {
            parts.append(f"  - {obs}");
        }
    }
    if slots.commands_run {
        parts.append("Commands run:");
        for cmd in slots.commands_run {
            parts.append(f"  - {cmd}");
        }
    }
    if slots.decisions {
        parts.append("Decisions:");
        for d in slots.decisions {
            parts.append(f"  - {d}");
        }
    }
    if slots.unresolved_questions {
        parts.append("Unresolved:");
        for q in slots.unresolved_questions {
            parts.append(f"  - {q}");
        }
    }
    return "\n".join(parts);
}


def _slot_entries(value: object) -> list[str] {
    if value is None {
        return [];
    }
    if isinstance(value, str) {
        return [value];
    }
    if isinstance(value, (list, tuple, set)) {
        seq: any = value;
        return [str(v) for v in seq];
    }
    return [str(value)];
}


impl coerce_compaction_slots(value: object) -> CompactionSlots {
    if isinstance(value, CompactionSlots) {
        return value;
    }
    raw: any = value;
    if isinstance(raw, str) {
        raw = json.loads(raw);
    }
    if isinstance(raw, dict) and len(raw) == 1 and "schema_object_wrapper" in raw {
        raw = raw["schema_object_wrapper"];
    }
    if not isinstance(raw, dict) {
        raise TypeError(
            "typed compaction expected CompactionSlots or a mapping, got "
            f"{type(value).__name__}"
        );
    }
    data = raw as dict[str, any];
    return CompactionSlots(
        files_read=_slot_entries(data.get("files_read")),
        key_observations=_slot_entries(data.get("key_observations")),
        commands_run=_slot_entries(data.get("commands_run")),
        decisions=_slot_entries(data.get("decisions")),
        unresolved_questions=_slot_entries(data.get("unresolved_questions"))
    );
}


impl extract_compaction_slots(
    model: object,
    previous_summary: str,
    items: list[dict[str, any]],
    parent_mt_run: object = None
) -> CompactionSlots {
    payload = json.dumps(
        {"previous_summary": previous_summary, "items": items},
        ensure_ascii=False,
        `default=str
    );
    comp_mt_run = MTRuntime(
        messages=[
            Message(role=MessageRole.SYSTEM, content=COMPACTION_EXTRACT_INSTRUCTIONS),
            Message(role=MessageRole.USER, content=payload)
        ],
        tools=[],
        resp_type=CompactionSlots,
        stream=False,
        call_params={"compaction_enabled": False},
        mtir=None,
        _call_kind="compaction"
    );
    extractor: any = model;
    result = extractor.invoke(comp_mt_run);
    if parent_mt_run is not None {
        parent: any = parent_mt_run;
        parent._usage_history.extend(comp_mt_run._usage_history);
    }
    return coerce_compaction_slots(result);
}
