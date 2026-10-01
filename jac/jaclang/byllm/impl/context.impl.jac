impl ContextGraph.mint_tool_result_id(turn: int, tool_name: str) -> str {
    self._tr_seq += 1;
    return f"tr_{turn}_{tool_name}_{self._tr_seq}";
}

impl ContextGraph.mint_message_id(turn: int, role: str) -> str {
    self._msg_seq += 1;
    return f"msg_{turn}_{role}_{self._msg_seq}";
}

impl ContextGraph.record_tool_result(
    tool_name: str,
    args: dict[str, any],
    full_content: str,
    turn: int = -1,
    call_id: str = "",
    media: bool = False
) -> ContextToolResult {
    if turn < 0 {
        self.turn_count += 1;
        turn = self.turn_count;
    }
    tr = ContextToolResult(
        tr_id=self.mint_tool_result_id(turn, tool_name),
        tool_name=tool_name,
        full_content=full_content,
        args=args,
        visible_content=full_content,
        produced_at=time.time(),
        produced_at_turn=turn,
        call_id=call_id,
        media=media
    );
    self ++> tr;
    return tr;
}

impl ContextGraph.record_message(
    role: str, content: str, tool_calls: list[dict[str, any]] | None = None
) -> ContextMessage {
    self.turn_count += 1;
    turn = self.turn_count;
    msg = ContextMessage(
        msg_id=self.mint_message_id(turn, role),
        role=role,
        turn=turn,
        content=content,
        token_estimate=estimate_tokens(content),
        tool_calls=tool_calls if tool_calls is not None else []
    );
    self ++> msg;
    return msg;
}

impl ContextGraph.tool_results -> list[ContextToolResult] {
    return [self-->][?:ContextToolResult];
}

impl ContextGraph.messages -> list[ContextMessage] {
    out = [self-->][?:ContextMessage];
    out.sort(key=lambda (m: ContextMessage) -> int { return m.turn; });
    return out;
}

impl ContextGraph.summaries -> list[ContextCompactionSummary] {
    out = [self-->][?:ContextCompactionSummary];
    out.sort(
        key=lambda (s: ContextCompactionSummary) -> int { return s.covers_turn_end; }
    );
    return out;
}

impl ContextPolicy.should_prune(
    cg: ContextGraph, mt_run: object, model: object
) -> bool {
    return False;
}

impl ContextPolicy.prune(cg: ContextGraph, mt_run: object, model: object) -> int {
    return 0;
}

impl ContextPolicy.compact(
    cg: ContextGraph, mt_run: object, model: object, keep_recent: int
) -> bool {
    return False;
}

impl ContextPolicy.materialize(cg: ContextGraph, mt_run: object) -> list[any] | None {
    return None;
}

impl Context.postinit -> None {
    if self.policy is not None {
        return;
    }
    try {
        import from jaclang.byllm.context_render { ContextRender }
        self.policy = ContextRender() as ContextPolicy;
    } except Exception {}
}

impl Context.get_or_create_graph -> ContextGraph {
    if self.graph is not None and self.persist {
        return self.graph;
    }
    cg = ContextGraph(session_id=self.session_id, created_at=time.time());
    self.graph = cg;
    return cg;
}

impl Context.reset -> None {
    self.graph = None;
    self._resolved_budget = 0;
}

impl Context.stats -> dict[str, any] {
    cg = self.graph
        if self.graph is not None
        else ContextGraph(session_id=self.session_id);
    out: dict[str, any] = {
        "session_id": self.session_id,
        "budget_tokens": self._resolved_budget or self.budget_tokens
    };
    out.update(graph_stats(cg));
    return out;
}

impl apply_eviction_decision(decision: EvictionDecision) -> int {
    tr = decision.tool_result;
    kind = decision.kind;
    if kind == "skip" or tr.truncation_state == "hard_cleared" {
        return 0;
    }
    before = estimate_tokens(tr.visible_content);
    if kind == "hard_clear" {
        tr.visible_content = decision.placeholder;
        tr.truncation_state = "hard_cleared";
    } elif kind == "soft_trim" {
        head = int(decision.head_chars);
        tail = int(decision.tail_chars);
        cur = str(tr.visible_content);
        if len(cur) > head + tail {
            tr.visible_content = cur[:head] + SOFT_TRIM_MARKER + cur[-tail:];
            if tr.truncation_state == "full" {
                tr.truncation_state = "head_tail";
            }
        }
    }
    after = estimate_tokens(tr.visible_content);
    return max(0, before - after);
}

impl estimate_tokens(text: str) -> int {
    return len(text) // 4;
}

impl distinctive_idents(text: str) -> set[str] {
    out: set[str] = set();
    if not text {
        return out;
    }
    for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.\-]{4,}", text) {
        ts = str(t).strip(".-_");
        if len(ts) < 6 {
            continue;
        }
        if ts[0].isupper() and ts[1:].islower() and ts.isalpha() {
            continue;
        }
        has_digit = False;
        for ch in ts {
            if ch.isdigit() {
                has_digit = True;
                break;
            }
        }
        if ("_" in ts)
            or (ts != ts.lower())
            or has_digit
            or ("-" in ts)
            or ("." in ts) {
            out.add(ts);
        }
    }
    return out;
}

impl lexical_tokens(text: str) -> set[str] {
    out: set[str] = set();
    for t in re.findall(r"[a-z0-9_]{3,}", text.lower()) {
        out.add(str(t));
    }
    return out;
}

impl lexical_set_cosine(a: set[str], b: set[str]) -> float {
    if not a or not b {
        return 0.0;
    }
    inter = len(a & b);
    if inter == 0 {
        return 0.0;
    }
    return inter / math.sqrt(len(a) * len(b));
}

impl cosine(a: list[float], b: list[float]) -> float {
    if not a or not b or len(a) != len(b) {
        return 0.0;
    }
    dot = 0.0;
    na = 0.0;
    nb = 0.0;
    for i in range(len(a)) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 or nb <= 0.0 {
        return 0.0;
    }
    return dot / (math.sqrt(na) * math.sqrt(nb));
}

impl unit(v: list[float]) -> list[float] {
    n = 0.0;
    for x in v {
        n += x * x;
    }
    if n <= 0.0 {
        return v;
    }
    norm = math.sqrt(n);
    return [x / norm for x in v];
}

impl cache_get(cache: object, key: str) -> any {
    store: any = cache;
    if key not in store {
        return None;
    }
    value: any = store.pop(key);
    store[key] = value;
    return value;
}

impl cache_put(cache: object, key: str, value: object, limit: int) {
    store: any = cache;
    store.pop(key, None);
    while limit > 0 and len(store) >= limit {
        store.pop(next(iter(store)));
    }
    store[key] = value;
}

impl cached_idents(text: str, cap: int = 20000) -> set[str] {
    clipped = text[:cap];
    key = f"{cap}:" + hashlib.sha1(clipped.encode("utf-8")).hexdigest();
    hit = cache_get(IDENT_CACHE, key);
    if hit is not None {
        return hit;
    }
    v = distinctive_idents(clipped);
    cache_put(IDENT_CACHE, key, v, TEXT_CACHE_MAX);
    return v;
}

impl cached_lextoks(text: str, cap: int = 20000) -> set[str] {
    clipped = text[:cap];
    key = f"{cap}:" + hashlib.sha1(clipped.encode("utf-8")).hexdigest();
    hit = cache_get(LEXTOK_CACHE, key);
    if hit is not None {
        return hit;
    }
    v = lexical_tokens(clipped);
    cache_put(LEXTOK_CACHE, key, v, TEXT_CACHE_MAX);
    return v;
}

impl graph_stats(cg: ContextGraph) -> dict[str, any] {
    out: dict[str, any] = {};
    for key in STAT_COUNTERS {
        out[key] = int(cg.stats.get(key, 0) or 0);
    }
    total = 0;
    visible_n = 0;
    hidden_n = 0;
    full_n = 0;
    trimmed_n = 0;
    cleared_n = 0;
    visible_tokens = 0;
    full_tokens = 0;
    for tr in cg.tool_results() {
        total += 1;
        if not tr.visible {
            hidden_n += 1;
            continue;
        }
        visible_n += 1;
        visible_tokens += estimate_tokens(tr.visible_content);
        full_tokens += estimate_tokens(tr.full_content);
        if tr.truncation_state == "hard_cleared" {
            cleared_n += 1;
        } elif tr.truncation_state == "full" {
            full_n += 1;
        } else {
            trimmed_n += 1;
        }
    }
    msgs = cg.messages();
    out["turn_count"] = cg.turn_count;
    out["messages"] = len(msgs);
    out["hidden_messages"] = len(
        [
            m
            for m in msgs
            if not m.visible
        ]
    );
    out["tool_results"] = total;
    out["visible_tool_results"] = visible_n;
    out["hidden_tool_results"] = hidden_n;
    out["full"] = full_n;
    out["trimmed"] = trimmed_n;
    out["hard_cleared"] = cleared_n;
    out["summaries"] = len(
        [
            s
            for s in cg.summaries()
            if s.visible
        ]
    );
    out["visible_tokens"] = visible_tokens;
    out["full_tokens"] = full_tokens;
    out["saved_tokens"] = max(0, full_tokens - visible_tokens);
    return out;
}

impl managed_tool_results(cg: ContextGraph) -> list[ContextToolResult] {
    seen: dict[str, int] = {};
    for tr in cg.tool_results() {
        if tr.call_id {
            seen[tr.call_id] = seen.get(tr.call_id, 0) + 1;
        }
    }
    return [
        tr
        for tr in cg.tool_results()
        if tr.call_id and not tr.media and seen.get(tr.call_id, 0) == 1
    ];
}

impl message_role(message: object) -> str {
    if isinstance(message, Message) {
        return str(message.role.value);
    }
    if isinstance(message, dict) {
        return str(message.get("role", "") or "");
    }
    role: any = message?.role;
    if role is None {
        return "";
    }
    return str(getattr(role, "value", role) or "");
}

impl message_content(message: object) -> any {
    if isinstance(message, dict) {
        return message.get("content");
    }
    return message?.content;
}

impl message_text(message: object) -> str {
    content: any = message_content(message);
    if content is None {
        return "";
    }
    if isinstance(content, str) {
        return content;
    }
    if not isinstance(content, (list, tuple)) {
        return str(content);
    }
    parts: list[str] = [];
    in_media = False;
    for part in content {
        text: str | None = None;
        if isinstance(part, str) {
            text = part;
        } elif isinstance(part, Text) {
            text = part.text;
        } elif isinstance(part, dict) and part.get("type") == "text" {
            text = str(part.get("text", "") or "");
        }
        if text is None {
            if not in_media {
                parts.append(MEDIA_TAG);
            }
            in_media = True;
            continue;
        }
        in_media = False;
        parts.append(text);
    }
    return "\n".join(parts);
}

def _tool_call_args(raw_args: object) -> dict[str, any] {
    if isinstance(raw_args, dict) {
        return raw_args as dict[str, any];
    }
    if isinstance(raw_args, str) and raw_args {
        try {
            parsed: any = json.loads(raw_args);
            if isinstance(parsed, dict) {
                return parsed as dict[str, any];
            }
        } except Exception {}
        return {"_raw": raw_args};
    }
    return {};
}

impl message_tool_calls(message: object) -> list[dict[str, any]] {
    raw: any = message?.tool_calls;
    if raw is None and isinstance(message, dict) {
        raw = message.get("tool_calls");
    }
    if not raw {
        return [];
    }
    out: list[dict[str, any]] = [];
    for tc in raw {
        tc_id = str(getattr(tc, "id", "") or "");
        tc_name = "";
        tc_args: dict[str, any] = {};
        fn: any = tc?.function;
        if fn is not None {
            tc_name = str(getattr(fn, "name", "") or "");
            tc_args = _tool_call_args(fn?.arguments);
        }
        if not tc_id and isinstance(tc, dict) {
            tc_id = str(tc.get("id", "") or "");
            fn_dict: any = tc.get("function") or {};
            if isinstance(fn_dict, dict) {
                tc_name = str(fn_dict.get("name", "") or "");
                tc_args = _tool_call_args(fn_dict.get("arguments"));
            }
        }
        if tc_id {
            out.append({"id": tc_id, "name": tc_name, "args": tc_args});
        }
    }
    return out;
}

impl tool_call_id_of(message: object) -> str {
    if isinstance(message, dict) {
        return str(message.get("tool_call_id", "") or "");
    }
    return str(getattr(message, "tool_call_id", "") or "");
}

impl with_content(message: object, content: str) -> object {
    if isinstance(message, ToolCallResultMsg) {
        return ToolCallResultMsg(
            content=content, tool_call_id=message.tool_call_id, name=message.name
        );
    }
    if isinstance(message, dict) {
        copied: dict[str, any] = dict(message);
        copied["content"] = content;
        return copied;
    }
    dumper: any = message?.model_dump;
    if callable(dumper) {
        dumped: any = dumper();
        if isinstance(dumped, dict) {
            dumped["content"] = content;
            return dumped;
        }
    }
    return message;
}

impl materialize_overlay(
    cg: ContextGraph, mt_run: object, appended: list[str]
) -> list[any] | None {
    tr_by_call: dict[str, ContextToolResult] = {};
    for tr in managed_tool_results(cg) {
        tr_by_call[tr.call_id] = tr;
    }
    hidden_ids: set[str] = set();
    for m in cg.messages() {
        if not m.visible {
            hidden_ids.add(m.msg_id);
        }
    }
    if not tr_by_call and not hidden_ids and not appended {
        return None;
    }
    raw_ids: any = mt_run?._ctx_msg_ids;
    msg_ids: dict[int, str] = (raw_ids as dict[int, str])
        if isinstance(raw_ids, dict)
        else {};
    raw_msgs: any = mt_run?.messages;
    source: list[any] = (raw_msgs as list[any]) if isinstance(raw_msgs, list) else [];
    summaries = [
        s
        for s in cg.summaries()
        if s.visible
    ];
    out: list[any] = [];
    summaries_emitted = False;
    first_user_seen = False;
    dropped_calls: set[str] = set();
    emitted_calls: set[str] = set();
    for message in source {
        role = message_role(message);
        if role == "tool" {
            call_id = tool_call_id_of(message);
            if call_id and call_id in dropped_calls {
                continue;
            }
            tr = tr_by_call.get(call_id) if call_id else None;
            if tr is None {
                out.append(message);
                continue;
            }
            if not tr.visible {
                if call_id in emitted_calls {
                    out.append(with_content(message, FOLDED_RESULT_NOTE));
                }
                continue;
            }
            if tr.visible_content == tr.full_content {
                out.append(message);
            } else {
                out.append(with_content(message, tr.visible_content));
            }
            continue;
        }
        protected = role == "system" or (role == "user" and not first_user_seen);
        if role == "user" {
            first_user_seen = True;
        }
        node_id = msg_ids.get(id(message));
        if node_id is not None and node_id in hidden_ids and not protected {
            for tc in message_tool_calls(message) {
                dropped_calls.add(str(tc["id"]));
            }
            if not summaries_emitted {
                summaries_emitted = True;
                for s in summaries {
                    out.append(
                        {
                            "role": "user",
                            "content": COMPACTION_TAG + "\n" + s.summary_text
                        }
                    );
                }
            }
            continue;
        }
        for tc in message_tool_calls(message) {
            emitted_calls.add(str(tc["id"]));
        }
        out.append(message);
    }
    if summaries and not summaries_emitted {
        insert_at = 0;
        while insert_at < len(out) and message_role(out[insert_at]) == "system" {
            insert_at += 1;
        }
        if insert_at < len(out) and message_role(out[insert_at]) == "user" {
            insert_at += 1;
        }
        summary_msgs: list[any] = [
            {"role": "user", "content": COMPACTION_TAG + "\n" + s.summary_text}
            for s in summaries
        ];
        out = out[:insert_at] + summary_msgs + out[insert_at:];
    }
    for text in appended {
        out.append({"role": "user", "content": text});
    }
    return out;
}
