impl Presence.has_result(tr: ContextToolResult) -> bool {
    return self.unrestricted or (tr.call_id in self.call_ids);
}

impl Presence.has_message(m: ContextMessage) -> bool {
    return self.unrestricted or (m.msg_id in self.msg_ids);
}

impl context_of(mt_run: object) -> Context | None {
    params: any = mt_run?.call_params;
    if not isinstance(params, dict) {
        return None;
    }
    ctx_obj: any = params.get("context");
    if isinstance(ctx_obj, Context) {
        return ctx_obj;
    }
    return None;
}

impl presence_of(mt_run: object) -> Presence {
    raw_msgs: any = mt_run?.messages;
    if not isinstance(raw_msgs, list) {
        return Presence(unrestricted=True, call_ids=set(), msg_ids=set());
    }
    raw_ids: any = mt_run?._ctx_msg_ids;
    ids: dict[int, str] = (raw_ids as dict[int, str])
        if isinstance(raw_ids, dict)
        else {};
    call_ids: set[str] = set();
    msg_ids: set[str] = set();
    for m in (raw_msgs as list[any]) {
        if message_role(m) == "tool" {
            cid = tool_call_id_of(m);
            if cid {
                call_ids.add(cid);
            }
            continue;
        }
        nid = ids.get(id(m));
        if nid is not None {
            msg_ids.add(nid);
        }
    }
    return Presence(unrestricted=False, call_ids=call_ids, msg_ids=msg_ids);
}

impl candidates_of(cg: ContextGraph, present: Presence) -> list[ContextToolResult] {
    return [
        tr
        for tr in managed_tool_results(cg)
        if tr.visible
            and tr.truncation_state != "hard_cleared"
            and present.has_result(tr)
    ];
}

impl visible_tool_tokens(cg: ContextGraph, present: Presence) -> int {
    vis = 0;
    for tr in managed_tool_results(cg) {
        if tr.visible and present.has_result(tr) {
            vis += estimate_tokens(str(tr.visible_content));
        }
    }
    return vis;
}

impl fill_tokens(cg: ContextGraph, present: Presence) -> int {
    total = 0;
    for m in cg.messages() {
        if m.visible and present.has_message(m) {
            total += m.token_estimate;
        }
    }
    for s in cg.summaries() {
        if s.visible {
            total += s.token_estimate;
        }
    }
    return total + visible_tool_tokens(cg, present);
}

impl current_turn_floor(cg: ContextGraph, present: Presence) -> int {
    floor_turn = -1;
    for m in cg.messages() {
        if m.visible
            and m.role == "assistant"
            and present.has_message(m)
            and m.turn > floor_turn {
            floor_turn = m.turn;
        }
    }
    return floor_turn;
}

impl bump_stat(cg: ContextGraph, key: str, n: int = 1) {
    cg.stats[key] = int(cg.stats.get(key, 0) or 0) + n;
}

impl scan_citations(cg: ContextGraph, cap: int = 20000) {
    trs = managed_tool_results(cg);
    asst = [
        m
        for m in cg.messages()
        if m.role == "assistant"
    ];
    if len(asst) <= cg.n_seen_asst {
        return;
    }
    for tr in trs {
        if tr.call_id not in cg.citation_idents {
            idents: list[str] = [];
            for ident in cached_idents(str(tr.full_content), cap) {
                idents.append(ident);
            }
            cg.citation_idents[tr.call_id] = idents;
        }
    }
    k = cg.n_seen_asst;
    while k < len(asst) {
        linked: dict[str, bool] = {};
        txt = str(asst[k].content or "");
        for tc in asst[k].tool_calls {
            try {
                txt += "\n" + json.dumps(tc);
            } except Exception {
                txt += "\n" + str(tc);
            }
        }
        if txt {
            cit_turn = asst[k].turn;
            for t in cached_idents(txt, cap) {
                sources: list[ContextToolResult] = [];
                for tr in trs {
                    if (
                        (tr.produced_at_turn < cit_turn)
                        and (t in (cg.citation_idents.get(tr.call_id) or []))
                    ) {
                        sources.append(tr);
                    }
                }
                if not sources {
                    continue;
                }
                w = 1.0 / len(sources);
                src = sources[0];
                for s_tr in sources {
                    if s_tr.produced_at < src.produced_at {
                        src = s_tr;
                    }
                }
                cg.citation_counts[src.call_id] = (
                    (cg.citation_counts.get(src.call_id) or 0.0) + w
                );
                ct_list = cg.citation_turns.get(src.call_id) or [];
                if cit_turn not in ct_list {
                    ct_list.append(cit_turn);
                    cg.citation_turns[src.call_id] = ct_list;
                    src.reference_count += 1;
                }
                src.last_referenced_at = time.time();
                if not linked.get(src.call_id) {
                    asst[k] +>:ReferencedBy(cite_turn=cit_turn):+> src;
                    linked[src.call_id] = True;
                }
            }
        }
        k += 1;
    }
    cg.n_seen_asst = len(asst);
}

impl active_call_ids(
    cg: ContextGraph, turns: int, cap: int = 20000
) -> dict[str, bool] {
    active: dict[str, bool] = {};
    if turns <= 0 {
        return active;
    }
    msgs = [
        m
        for m in cg.messages()
        if m.role == "assistant"
    ];
    if not msgs {
        return active;
    }
    floor_turn = cg.turn_count - turns;
    recent_idents: set[str] = set();
    for m in msgs {
        if m.turn <= floor_turn {
            continue;
        }
        txt = str(m.content or "");
        for tc in m.tool_calls {
            try {
                txt += "\n" + json.dumps(tc);
            } except Exception {}
        }
        for t in cached_idents(txt, cap) {
            recent_idents.add(t);
        }
    }
    if not recent_idents {
        return active;
    }
    for tr in managed_tool_results(cg) {
        if len(cached_idents(str(tr.full_content), cap) & recent_idents) > 0 {
            active[tr.call_id] = True;
        }
    }
    return active;
}

impl focus_texts(cg: ContextGraph, present: Presence) -> dict[str, str] {
    last_user = "";
    last_asst = "";
    first_user = "";
    latest_user_turn = -1;
    latest_asst_turn = -1;
    earliest_user_turn = -1;
    for m in cg.messages() {
        if not m.visible or not present.has_message(m) {
            continue;
        }
        if m.role == "user" {
            if m.turn >= latest_user_turn {
                last_user = m.content;
                latest_user_turn = m.turn;
            }
            if earliest_user_turn < 0 or m.turn < earliest_user_turn {
                first_user = m.content;
                earliest_user_turn = m.turn;
            }
        } elif m.role == "assistant" and m.turn >= latest_asst_turn {
            last_asst = m.content;
            latest_asst_turn = m.turn;
        }
    }
    anchor = first_user;
    focus = (last_user + "\n" + last_asst).strip();
    if not focus {
        focus = anchor;
    }
    return {"anchor": anchor, "focus": focus};
}

impl evicted_handle(tr: ContextToolResult) -> str {
    args_hint = "";
    try {
        args_hint = json.dumps(tr.args)[:120] if tr.args else "";
    } except Exception {
        args_hint = "";
    }
    call_desc = f"{tr.tool_name}({args_hint})" if args_hint else str(tr.tool_name);
    size = estimate_tokens(str(tr.full_content));
    return (
        f"[evicted from context: {call_desc} result from turn "
        + f"{tr.produced_at_turn}, ~{size} tokens. Re-run this call if you "
        + "need its contents.]"
    );
}

impl restore_if_trimmed(tr: ContextToolResult) -> bool {
    if tr.truncation_state != "full" {
        tr.visible_content = str(tr.full_content);
        tr.truncation_state = "full";
        return True;
    }
    return False;
}

impl enforce_budget(
    ids: list[str], costs: dict[str, int], priority: dict[str, float], budget: int
) -> dict[str, bool] {
    keep: dict[str, bool] = {};
    total = 0;
    for i in ids {
        keep[i] = True;
        total += costs.get(i) or 0;
    }
    if total <= budget or len(ids) <= 1 {
        return keep;
    }
    order = [i for i in ids];
    order.sort(key=lambda (x: str) -> float { return priority.get(x) or 0.0; });
    k = 0;
    while total > budget and k < (len(order) - 1) {
        vic = order[k];
        keep[vic] = False;
        total -= costs.get(vic) or 0;
        k += 1;
    }
    return keep;
}

impl stub_reserve(n_candidates: int, budget: int, stub_chars: int) -> int {
    stub = (stub_chars // 4) + 8;
    reserve = n_candidates * stub;
    return max(budget // 4, budget - reserve);
}

def _call_embedding(model_name: str, text: str) -> list[float] {
    raw: any = litellm.embedding(model=model_name, input=text, timeout=EMBED_TIMEOUT);
    data: any = raw["data"];
    first: any = data[0];
    emb: any = first["embedding"];
    return [float(v) for v in emb];
}

impl embed_text(model_name: str, text: str, clip: int) -> list[float] {
    clipped = text[:clip];
    key = model_name + ":" + hashlib.sha1(clipped.encode("utf-8")).hexdigest();
    cached = EMBED_CACHE.get(key);
    if cached is not None {
        return cached;
    }
    vec: list[float] = [];
    if not EMBED_STATE.get("disabled") {
        try {
            vec = _call_embedding(model_name, clipped);
        } except Exception as e {
            try {
                vec = _call_embedding(model_name, clipped);
            } except Exception {
                logger.warning(
                    "byLLM ContextRender: embedding call failed twice "
                    f"({type(e).__name__}: {str(e)[:120]}); relevance is lexical "
                    "for the rest of this process"
                );
                EMBED_STATE["disabled"] = True;
                vec = [];
            }
        }
    }
    EMBED_CACHE[key] = vec;
    return vec;
}

impl message_item_text(m: ContextMessage) -> str {
    text = str(m.content or "");
    if text or not m.tool_calls {
        return text;
    }
    descs: list[str] = [];
    for tc in m.tool_calls {
        name = str(tc.get("name", "") or "?");
        args = "";
        try {
            args = json.dumps(tc.get("args", {}), `default=str);
        } except Exception {
            args = str(tc.get("args", {}));
        }
        descs.append(f"{name}({args})");
    }
    return "called: " + "; ".join(descs);
}

impl ContextRender._resolve_budget(
    ctx: Context | None, mt_run: object, model: object
) -> int {
    budget = 0;
    if ctx is not None and ctx.budget_tokens > 0 {
        budget = int(ctx.budget_tokens);
    } else {
        cfg_budget = 0;
        cfg_ratio = 0.5;
        try {
            cfg = get_byllm_config().get_context_config();
            cfg_budget = int(cfg.get("budget_tokens", 0) or 0);
            cfg_ratio = float(cfg.get("budget_ratio", 0.5) or 0.0);
        } except Exception as e {
            logger.warning(
                f"byLLM ContextRender: [plugins.byllm.context] could not be read ({e}); "
                "using the built-in defaults"
            );
        }
        if cfg_budget > 0 {
            budget = cfg_budget;
        } else {
            ratio = float(ctx.budget_ratio) if ctx is not None else cfg_ratio;
            window = 0;
            window_fn: any = model?._get_ctx_window;
            if callable(window_fn) {
                params: any = mt_run?.call_params;
                try {
                    window = int(
                        window_fn(params if isinstance(params, dict) else {}) or 0
                    );
                } except Exception {
                    window = 0;
                }
            }
            budget = int(window * ratio) if window > 0 else 0;
        }
    }
    if ctx is not None {
        ctx._resolved_budget = budget;
    }
    return budget;
}

impl ContextRender._resolve_embed_model -> str {
    if self.embed_model {
        return self.embed_model;
    }
    try {
        return str(
            get_byllm_config().get_context_config().get("embed_model", "") or ""
        );
    } except Exception {
        return "";
    }
}

impl ContextRender.embed(text: str) -> list[float] {
    name = self._resolve_embed_model();
    if not name {
        return [];
    }
    return embed_text(name, text, self.embed_chars);
}

impl ContextRender.focus_query_vec(anchor: str, focus: str) -> list[float] {
    av = unit(self.embed(anchor)) if anchor else [];
    fv = unit(self.embed(focus)) if focus else [];
    if not av {
        return fv;
    }
    if not fv or len(av) != len(fv) {
        return av;
    }
    w = self.anchor_w;
    return [w * av[i] + (1.0 - w) * fv[i] for i in range(len(av))];
}

impl ContextRender.base_score(age: int, cmass: float, rel: float) -> float {
    c = cmass;
    if self.reuse_cap >= 0.0 and c > self.reuse_cap {
        c = self.reuse_cap;
    }
    return (
        self.w_recency * math.exp(-self.recency_decay * float(age))
        + self.w_reuse * (1.0 - math.exp(-self.reuse_saturation * c))
        + self.w_relevance * (rel if rel > 0.0 else 0.0)
    );
}

impl ContextRender.score_pass(
    cg: ContextGraph, candidates: list[ContextToolResult], present: Presence
) -> ScorePass {
    ft = focus_texts(cg, present);
    anchor_text = ft["anchor"];
    focus_text = ft["focus"];
    trim_tokens: set[str] = lexical_tokens(focus_text) if focus_text else set();
    goal_vec = self.focus_query_vec(anchor_text, focus_text);
    use_embeddings = len(goal_vec) > 0;
    valid: list[ContextToolResult] = [];
    vecs: list[list[float]] = [];
    tokensets: list[set[str]] = [];
    if use_embeddings {
        embed_failed = False;
        tmp_valid: list[ContextToolResult] = [];
        tmp_vecs: list[list[float]] = [];
        for tr in candidates {
            e = self.embed(str(tr.full_content));
            if not e {
                embed_failed = True;
                break;
            }
            tmp_valid.append(tr);
            tmp_vecs.append(e);
        }
        if embed_failed {
            use_embeddings = False;
        } else {
            valid = tmp_valid;
            vecs = tmp_vecs;
        }
    }
    if not use_embeddings {
        valid = [];
        for tr in candidates {
            valid.append(tr);
            tokensets.append(
                cached_lextoks(str(tr.full_content), self.score_content_cap)
            );
        }
    }
    sub_goal_tokens: set[str] = (
        (lexical_tokens(anchor_text) | lexical_tokens(focus_text))
            if not use_embeddings
            else set()
    );
    n = len(valid);
    rel: list[float] = [];
    base: list[float] = [];
    for i in range(n) {
        r = cosine(vecs[i], goal_vec)
            if use_embeddings
            else lexical_set_cosine(tokensets[i], sub_goal_tokens);
        rel.append(r);
        age = cg.turn_count - int(valid[i].produced_at_turn);
        cmass = float(cg.citation_counts.get(valid[i].call_id) or 0.0);
        b = self.base_score(age, cmass, r);
        if self.select_hysteresis > 0.0 and valid[i].truncation_state == "full" {
            b = b + self.select_hysteresis;
        }
        base.append(b);
    }
    return ScorePass(
        candidates=valid,
        use_embeddings=use_embeddings,
        vecs=vecs,
        tokensets=tokensets,
        rel=rel,
        base=base,
        trim_tokens=trim_tokens,
        anchor_text=anchor_text,
        focus_text=focus_text
    );
}

impl ContextRender.candidate_scores(
    cg: ContextGraph, mt_run: object
) -> dict[str, float] {
    present = presence_of(mt_run);
    sp = self.score_pass(cg, candidates_of(cg, present), present);
    out: dict[str, float] = {};
    for i in range(len(sp.candidates)) {
        out[sp.candidates[i].tr_id] = sp.base[i];
    }
    return out;
}

impl ContextRender.keep_cost(tr: ContextToolResult) -> int {
    if self.append_restore and tr.truncation_state != "full" {
        return (
            estimate_tokens(str(tr.full_content))
            + estimate_tokens(str(tr.visible_content))
        );
    }
    return estimate_tokens(str(tr.full_content));
}

impl ContextRender.tiered_kind(
    tr: ContextToolResult, current_floor: int, kind: str
) -> str {
    if self.protect_current_turn and (tr.produced_at_turn > current_floor) {
        return "soft_trim";
    }
    return kind;
}

impl ContextRender.smart_trim(
    tr: ContextToolResult, query_tokens: set[str], cg: ContextGraph
) -> bool {
    if tr.truncation_state == "hard_cleared" {
        return False;
    }
    content = str(tr.full_content);
    total = self.trim_chars;
    if len(content) <= total {
        return False;
    }
    if (
        self.sticky_trim
        and tr.truncation_state != "full"
        and tr.visible_content
        and len(str(tr.visible_content)) <= total
    ) {
        return False;
    }
    digest = hashlib.sha1(content.encode("utf-8", "ignore")).hexdigest()[:16];
    memo_key = f"trim:{tr.tr_id}:{digest}:{total}";
    if self.memo_trim {
        memo = cg.trim_memo.get(memo_key);
        if memo {
            tr.visible_content = memo;
            if tr.truncation_state == "full" {
                tr.truncation_state = "head_tail";
            }
            return True;
        }
    }
    head_budget = total // 3;
    tail_budget = total // 3;
    mid_budget = total - head_budget - tail_budget;
    lines = content.split("\n");
    nlines = len(lines);
    hi = 0;
    hchars = 0;
    while hi < nlines and hchars < head_budget {
        hchars += len(lines[hi]) + 1;
        hi += 1;
    }
    ti = nlines;
    tchars = 0;
    while ti > hi and tchars < tail_budget {
        tchars += len(lines[ti - 1]) + 1;
        ti -= 1;
    }
    if ti <= hi {
        tr.visible_content = (
            content[:head_budget] + "\n" + TRIM_MARKER + "\n" + content[-tail_budget:]
        );
        if self.memo_trim {
            cg.trim_memo[memo_key] = str(tr.visible_content);
        }
        if tr.truncation_state == "full" {
            tr.truncation_state = "head_tail";
        }
        return True;
    }
    mid_idxs = [i for i in range(hi, ti)];
    mid_score: dict[int, float] = {};
    for idx in mid_idxs {
        mid_score[idx] = (
            lexical_set_cosine(query_tokens, lexical_tokens(str(lines[idx])))
                if query_tokens
                else 0.0
        );
    }
    ranked = sorted(
        mid_idxs,
        key=lambda (i: int) -> float { return mid_score.get(i, 0.0) or 0.0; },
        reverse=True
    );
    keep_mid: dict[int, bool] = {};
    mchars = 0;
    for idx in ranked {
        if (mid_score.get(idx, 0.0) or 0.0) <= 0.0 {
            break;
        }
        c = len(lines[idx]) + 1;
        if mchars + c > mid_budget {
            continue;
        }
        keep_mid[idx] = True;
        mchars += c;
    }
    out: list[str] = [];
    for idx in range(0, hi) {
        out.append(str(lines[idx]));
    }
    prev_kept = hi - 1;
    for idx in range(hi, ti) {
        if keep_mid.get(idx) {
            if idx > prev_kept + 1 {
                out.append(TRIM_MARKER);
            }
            out.append(str(lines[idx]));
            prev_kept = idx;
        }
    }
    if prev_kept < ti - 1 {
        out.append(TRIM_MARKER);
    }
    for idx in range(ti, nlines) {
        out.append(str(lines[idx]));
    }
    tr.visible_content = "\n".join(out);
    if self.memo_trim {
        cg.trim_memo[memo_key] = str(tr.visible_content);
    }
    if tr.truncation_state == "full" {
        tr.truncation_state = "head_tail";
    }
    return True;
}

impl ContextRender.micro_stub(
    tr: ContextToolResult, cg: ContextGraph, target: int = 0
) -> bool {
    if tr.truncation_state == "hard_cleared" {
        return False;
    }
    content = str(tr.full_content);
    if len(content) <= self.micro_stub_min {
        return False;
    }
    total = (
        max(self.micro_stub_min, target)
            if target > 0
            else min(self.micro_stub_chars, max(self.micro_stub_min, len(content) // 2))
    );
    if (
        self.sticky_trim
        and tr.truncation_state != "full"
        and tr.visible_content
        and len(str(tr.visible_content)) <= total
    ) {
        return False;
    }
    digest = hashlib.sha1(content.encode("utf-8", "ignore")).hexdigest()[:16];
    memo_key = f"micro:{tr.tr_id}:{digest}:{total}";
    if self.memo_trim {
        memo = cg.trim_memo.get(memo_key);
        if memo {
            tr.visible_content = memo;
            if tr.truncation_state == "full" {
                tr.truncation_state = "head_tail";
            }
            return True;
        }
    }
    lines = content.split("\n");
    nlines = len(lines);
    idsets: dict[int, set[str]] = {};
    for i in range(0, nlines) {
        idsets[i] = distinctive_idents(str(lines[i]));
    }
    ranked = sorted(
        [i for i in range(0, nlines)],
        key=lambda (i: int) -> float { return (
            float(len(idsets.get(i) or set())) / float(len(str(lines[i])) + 1)
        ); },
        reverse=True
    );
    keep: dict[int, bool] = {};
    seen: set[str] = set();
    used = 0;
    if nlines > 0 and (len(str(lines[0])) + 1) <= total {
        keep[0] = True;
        used = len(str(lines[0])) + 1;
        seen = seen | (idsets.get(0) or set());
    }
    for i in ranked {
        if keep.get(i) {
            continue;
        }
        ids = idsets.get(i) or set();
        if not (ids - seen) {
            continue;
        }
        c = len(str(lines[i])) + 1;
        if used + c > total {
            continue;
        }
        keep[i] = True;
        used += c;
        seen = seen | ids;
    }
    if not keep {
        return False;
    }
    out: list[str] = [];
    prev = -1;
    for i in range(0, nlines) {
        if keep.get(i) {
            if i > prev + 1 {
                out.append(TRIM_MARKER);
            }
            out.append(str(lines[i]));
            prev = i;
        }
    }
    if prev < nlines - 1 {
        out.append(TRIM_MARKER);
    }
    tr.visible_content = "\n".join(out);
    if self.memo_trim {
        cg.trim_memo[memo_key] = str(tr.visible_content);
    }
    if tr.truncation_state == "full" {
        tr.truncation_state = "head_tail";
    }
    return True;
}

impl ContextRender.apply_eviction(
    tr: ContextToolResult, kind: str, query_tokens: set[str], cg: ContextGraph
) -> str {
    if kind == "soft_trim" {
        if self.smart_trim(tr, query_tokens, cg) {
            return "trim";
        }
        if (
            tr.truncation_state == "full"
            and len(str(tr.full_content)) <= self.trim_chars
            and self.micro_stub(tr, cg)
        ) {
            return "stub";
        }
        return "";
    }
    placeholder = HARD_CLEAR_PLACEHOLDER;
    k = kind;
    if k == "micro_stub" {
        if self.micro_stub(tr, cg) {
            return "stub";
        }
        k = "hard_clear";
        placeholder = evicted_handle(tr);
    }
    apply_eviction_decision(
        EvictionDecision(tool_result=tr, kind=k, placeholder=placeholder)
    );
    return "clear";
}

impl ContextRender.clear_to_budget(
    cg: ContextGraph, budget_tokens: int, present: Presence
) -> int {
    if budget_tokens <= 0 {
        return 0;
    }
    vis = visible_tool_tokens(cg, present);
    if vis <= budget_tokens {
        return 0;
    }
    evicted = [
        tr
        for tr in managed_tool_results(cg)
        if tr.visible
            and present.has_result(tr)
            and tr.truncation_state in ["head_tail", "compacted"]
    ];
    evicted.sort(
        key=lambda (tr: ContextToolResult) -> float { return tr.produced_at; }
    );
    cleared = 0;
    for tr in evicted {
        if vis <= budget_tokens {
            break;
        }
        before = estimate_tokens(str(tr.visible_content));
        apply_eviction_decision(
            EvictionDecision(
                tool_result=tr, kind="hard_clear", placeholder=HARD_CLEAR_PLACEHOLDER
            )
        );
        vis -= before - estimate_tokens(str(tr.visible_content));
        cleared += 1;
    }
    return cleared;
}

impl ContextRender.should_prune(
    cg: ContextGraph, mt_run: object, model: object
) -> bool {
    budget = self._resolve_budget(context_of(mt_run), mt_run, model);
    if budget <= 0 {
        return False;
    }
    present = presence_of(mt_run);
    return float(visible_tool_tokens(cg, present)) > float(budget) * self.high_water;
}

impl ContextRender.prune(cg: ContextGraph, mt_run: object, model: object) -> int {
    budget = self._resolve_budget(context_of(mt_run), mt_run, model);
    if budget <= 0 {
        return 0;
    }
    present = presence_of(mt_run);
    candidates = candidates_of(cg, present);
    if not candidates {
        return 0;
    }
    scan_citations(cg, self.score_content_cap);
    sp = self.score_pass(cg, candidates, present);
    valid = sp.candidates;
    base = sp.base;
    n = len(valid);
    selected_idx: dict[int, bool] = {};
    used = 0;
    order = [i for i in range(n)];
    order.sort(
        key=lambda (i: int) -> tuple[float, int] { return (valid[i].produced_at, i); },
        reverse=True
    );
    grace_floor = cg.turn_count - self.grace_turns;
    grace_cap = int(float(budget) * self.pin_budget_frac);
    newest_kept = False;
    for i in order {
        if valid[i].produced_at_turn > grace_floor {
            cost = self.keep_cost(valid[i]);
            if (not newest_kept) or (used + cost <= grace_cap) {
                selected_idx[i] = True;
                used += cost;
                newest_kept = True;
            }
        }
    }
    pinned = 0;
    for i in order {
        if pinned >= self.pin_recent {
            break;
        }
        if selected_idx.get(i) {
            continue;
        }
        cost = self.keep_cost(valid[i]);
        if used + cost <= budget {
            selected_idx[i] = True;
            used += cost;
            pinned += 1;
        }
    }
    ws = active_call_ids(cg, self.working_set_turns, self.score_content_cap);
    if ws {
        for i in order {
            if selected_idx.get(i) or (not ws.get(str(valid[i].call_id))) {
                continue;
            }
            cost = self.keep_cost(valid[i]);
            if used + cost <= budget {
                selected_idx[i] = True;
                used += cost;
            }
        }
    }
    while True {
        best = -1;
        best_score = -1.0e9;
        for i in range(n) {
            if selected_idx.get(i) {
                continue;
            }
            cost = self.keep_cost(valid[i]);
            if used + cost > budget {
                continue;
            }
            maxsim = 0.0;
            for j in selected_idx.keys() {
                s = cosine(sp.vecs[i], sp.vecs[j])
                    if sp.use_embeddings
                    else lexical_set_cosine(sp.tokensets[i], sp.tokensets[j]);
                if s > maxsim {
                    maxsim = s;
                }
            }
            score = base[i] - self.mmr_lambda * maxsim;
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        if best < 0 {
            break;
        }
        selected_idx[best] = True;
        used += self.keep_cost(valid[best]);
    }
    sel_ids: list[str] = [];
    sel_cost: dict[str, int] = {};
    sel_prio: dict[str, float] = {};
    for i in selected_idx.keys() {
        tid = str(valid[i].tr_id);
        sel_ids.append(tid);
        sel_cost[tid] = self.keep_cost(valid[i]);
        sel_prio[tid] = base[i];
    }
    stub_chars = self.micro_stub_chars
        if self.stub_overflow == "clear"
        else self.trim_chars;
    eff_budget = stub_reserve(len(candidates), max(1, budget), stub_chars);
    survives = enforce_budget(sel_ids, sel_cost, sel_prio, eff_budget);
    keep_ids: dict[str, bool] = {};
    kept_total = 0;
    for i in selected_idx.keys() {
        tid = str(valid[i].tr_id);
        if survives.get(tid) {
            keep_ids[tid] = True;
            kept_total += sel_cost.get(tid) or 0;
        }
    }
    dropped: list[str] = [];
    for i in range(n) {
        if not keep_ids.get(valid[i].tr_id) {
            dropped.append(str(valid[i].tr_id));
        }
    }
    dropped.sort(key=lambda (x: str) -> float { return sel_prio.get(x) or -1.0e9; });
    esc: dict[str, bool] = {};
    if self.stub_overflow == "clear" {
        stub_t = (self.micro_stub_chars // 4) + 8;
        total_v = kept_total + (len(dropped) * stub_t);
        k = 0;
        while total_v > budget and k < len(dropped) {
            esc[dropped[k]] = True;
            total_v -= stub_t - 25;
            k += 1;
        }
    }
    if self.evict_min_tokens > 0 or self.evict_min_gain_ratio > 0.0 {
        removed_gate = 0;
        restores_gate = 0;
        earliest_gate = -1.0;
        for tr in candidates {
            is_keep = bool(keep_ids.get(tr.tr_id));
            is_full = tr.truncation_state == "full";
            if is_keep == is_full {
                continue;
            }
            if is_keep {
                restores_gate += 1;
            } else {
                removed_gate += estimate_tokens(str(tr.full_content));
            }
            if earliest_gate < 0.0 or tr.produced_at < earliest_gate {
                earliest_gate = tr.produced_at;
            }
        }
        suffix_gate = 0;
        for tr in candidates {
            if earliest_gate >= 0.0 and tr.produced_at > earliest_gate {
                suffix_gate += estimate_tokens(str(tr.visible_content));
            }
        }
        if restores_gate == 0
            and (
                removed_gate < self.evict_min_tokens
                or removed_gate < int(self.evict_min_gain_ratio * float(suffix_gate))
            ) {
            return 0;
        }
    }
    fill_ratio = float(fill_tokens(cg, present)) / float(budget);
    kind = "micro_stub"
        if (self.allow_hard_clear and fill_ratio >= 0.5)
        else "soft_trim";
    current_floor = current_turn_floor(cg, present);
    before: dict[str, str] = {};
    for tr in candidates {
        before[tr.tr_id] = str(tr.visible_content);
    }
    restores = 0;
    trims = 0;
    stubs = 0;
    clears = 0;
    for tr in candidates {
        if keep_ids.get(tr.tr_id) {
            if self.append_restore {
                if tr.truncation_state != "full" {
                    cg.append_restored[tr.call_id] = True;
                }
            } elif restore_if_trimmed(tr) {
                restores += 1;
            }
            continue;
        }
        if self.append_restore and cg.append_restored.get(tr.call_id) {
            cg.append_restored[tr.call_id] = False;
        }
        k2 = "micro_stub"
            if esc.get(str(tr.tr_id))
            else self.tiered_kind(tr, current_floor, kind);
        outcome = self.apply_eviction(tr, k2, sp.trim_tokens, cg);
        if outcome == "trim" {
            trims += 1;
        } elif outcome == "stub" {
            stubs += 1;
        } elif outcome == "clear" {
            clears += 1;
        }
    }
    clears += self.clear_to_budget(cg, budget, present);
    changed = 0;
    for tr in candidates {
        if str(tr.visible_content) != before.get(tr.tr_id, "") {
            changed += 1;
        }
    }
    bump_stat(cg, "prunes", 1);
    bump_stat(cg, "restores", restores);
    bump_stat(cg, "trims", trims);
    bump_stat(cg, "stubs", stubs);
    bump_stat(cg, "clears", clears);
    logger.debug(
        f"byLLM ContextRender: pass at turn {cg.turn_count}, budget {budget}, "
        f"{len(candidates)} candidates, {len(keep_ids)} kept, {changed} changed "
        f"(restores {restores}, trims {trims}, stubs {stubs}, clears {clears})"
    );
    return changed;
}

impl ContextRender.materialize(cg: ContextGraph, mt_run: object) -> list[any] | None {
    appended: list[str] = [];
    if self.append_restore {
        present = presence_of(mt_run);
        restored = [
            tr
            for tr in managed_tool_results(cg)
            if tr.visible
                and present.has_result(tr)
                and cg.append_restored.get(tr.call_id)
        ];
        restored.sort(
            key=lambda (tr: ContextToolResult) -> float { return tr.produced_at; }
        );
        for tr in restored {
            args_json = "";
            try {
                args_json = json.dumps(tr.args, `default=str) if tr.args else "";
            } except Exception {
                args_json = "";
            }
            appended.append(
                f"[Restored tool result {tr.tool_name}({args_json})]\n{tr.full_content}"
            );
        }
    }
    return materialize_overlay(cg, mt_run, appended);
}

impl ContextRender.compact(
    cg: ContextGraph, mt_run: object, model: object, keep_recent: int
) -> bool {
    present = presence_of(mt_run);
    msgs = [
        m
        for m in cg.messages()
        if m.visible and present.has_message(m)
    ];
    first_user_id = "";
    for m in msgs {
        if m.role == "user" {
            first_user_id = m.msg_id;
            break;
        }
    }
    cut = cg.turn_count + 1;
    if keep_recent > 0 {
        rounds = 0;
        found = False;
        i = len(msgs) - 1;
        while i >= 0 {
            if msgs[i].role == "assistant" {
                rounds += 1;
                if rounds >= keep_recent {
                    cut = msgs[i].turn;
                    found = True;
                    break;
                }
            }
            i -= 1;
        }
        if not found {
            return False;
        }
    }
    covered_msgs: list[ContextMessage] = [];
    for m in msgs {
        if m.turn >= cut or m.role == "system" {
            continue;
        }
        if self.compact_keep_first_user and m.msg_id == first_user_id {
            continue;
        }
        covered_msgs.append(m);
    }
    covered_trs = [
        tr
        for tr in managed_tool_results(cg)
        if tr.visible and present.has_result(tr) and tr.produced_at_turn < cut
    ];
    if not covered_msgs and not covered_trs {
        return False;
    }
    items: list[dict[str, any]] = [];
    for m in covered_msgs {
        items.append({"turn": m.turn, "role": m.role, "content": message_item_text(m)});
    }
    for tr in covered_trs {
        items.append(
            {
                "turn": tr.produced_at_turn,
                "role": f"tool:{tr.tool_name}",
                "content": str(tr.full_content)
            }
        );
    }
    items.sort(key=lambda (x: dict[str, any]) -> int { return int(x["turn"]); });
    prior = [
        s
        for s in cg.summaries()
        if s.visible
    ];
    previous_summary = prior[-1].summary_text if prior else "";
    summary_text = compaction_summary_text(model, previous_summary, items, mt_run);
    turns = [m.turn for m in covered_msgs]
        + [tr.produced_at_turn for tr in covered_trs];
    summary = ContextCompactionSummary(
        generated_at=time.time(),
        covers_turn_start=min(turns),
        covers_turn_end=max(turns),
        summary_text=summary_text,
        dropped_message_count=len(covered_msgs),
        dropped_tool_result_count=len(covered_trs),
        token_estimate=estimate_tokens(summary_text)
    );
    cg ++> summary;
    for m in covered_msgs {
        m.visible = False;
    }
    for tr in covered_trs {
        tr.visible = False;
    }
    for s in prior {
        s.visible = False;
    }
    bump_stat(cg, "compactions", 1);
    logger.info(
        f"byLLM ContextRender: folded turns {min(turns)}-{max(turns)} "
        f"({len(covered_msgs)} messages, {len(covered_trs)} tool results) into a "
        f"{COMPACTION_TAG} block"
    );
    return True;
}
