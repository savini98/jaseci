import hashlib;
import json;
import math;
import time;
import from jaclang.byllm._optdeps.litellm { litellm }
import from jaclang.byllm._optdeps.loguru { logger }
import from jaclang.byllm.config_loader { get_byllm_config }
import from jaclang.byllm.context {
    COMPACTION_TAG,
    Context,
    ContextCompactionSummary,
    ContextGraph,
    ContextMessage,
    ContextPolicy,
    ContextToolResult,
    EvictionDecision,
    ReferencedBy,
    apply_eviction_decision,
    cached_idents,
    cached_lextoks,
    cosine,
    distinctive_idents,
    estimate_tokens,
    lexical_set_cosine,
    lexical_tokens,
    managed_tool_results,
    materialize_overlay,
    message_role,
    tool_call_id_of,
    unit
}
import from jaclang.byllm.exceptions { CompactionNotEffectiveError }
import from jaclang.byllm.typed_compaction {
    extract_compaction_slots,
    render_compaction_slots
}

glob TRIM_MARKER: str = "...[relevance-trimmed]...",
     HARD_CLEAR_PLACEHOLDER: str = "[tool result evicted by context policy - re-run the tool if needed]",
     NOTHING_TO_FOLD: str = "ContextRender compaction found nothing to fold - increase ctx_window, lower keep_recent_iterations, or switch to a model with a larger context window.",
     EMBED_TIMEOUT: float = 60.0,
     EMBED_CACHE: dict[str, list[float]] = {},
     EMBED_STATE: dict[str, bool] = {};

obj Presence {
    has unrestricted: bool,
        call_ids: set[str],
        msg_ids: set[str];

    def has_result(tr: ContextToolResult) -> bool;
    def has_message(m: ContextMessage) -> bool;
}

obj ScorePass {
    has candidates: list[ContextToolResult],
        use_embeddings: bool,
        vecs: list[list[float]],
        tokensets: list[set[str]],
        rel: list[float],
        base: list[float],
        trim_tokens: set[str],
        anchor_text: str,
        focus_text: str;
}

obj ContextRender(ContextPolicy) {
    has embed_model: str = "",
        anchor_w: float = 0.4,
        grace_turns: int = 3,
        pin_recent: int = 2,
        pin_budget_frac: float = 0.6,
        working_set_turns: int = 3,
        w_recency: float = 1.0,
        w_reuse: float = 1.0,
        w_relevance: float = 1.0,
        recency_decay: float = 0.3,
        reuse_saturation: float = 0.2,
        reuse_cap: float = -1.0,
        mmr_lambda: float = 0.5,
        trim_chars: int = 3000,
        micro_stub_chars: int = 150,
        micro_stub_min: int = 100,
        sticky_trim: bool = True,
        memo_trim: bool = True,
        stub_overflow: str = "keep",
        allow_hard_clear: bool = False,
        protect_current_turn: bool = True,
        high_water: float = 1.0,
        select_hysteresis: float = 0.0,
        evict_min_tokens: int = 0,
        evict_min_gain_ratio: float = 0.0,
        append_restore: bool = False,
        score_content_cap: int = 20000,
        embed_chars: int = 8000,
        compact_keep_first_user: bool = True,
        compact_full_ratio: float = 0.5;

    override def should_prune(cg: ContextGraph, mt_run: object, model: object) -> bool;
    override def prune(cg: ContextGraph, mt_run: object, model: object) -> int;
    override def compact(
        cg: ContextGraph, mt_run: object, model: object, keep_recent: int
    ) -> bool;

    override def materialize(cg: ContextGraph, mt_run: object) -> list[any] | None;
    def _resolve_budget(ctx: Context | None, mt_run: object, model: object) -> int;
    def _resolve_embed_model -> str;
    def embed(text: str) -> list[float];
    def focus_query_vec(anchor: str, focus: str) -> list[float];
    def base_score(age: int, cmass: float, rel: float) -> float;
    def score_pass(
        cg: ContextGraph, candidates: list[ContextToolResult], present: Presence
    ) -> ScorePass;

    def candidate_scores(cg: ContextGraph, mt_run: object) -> dict[str, float];
    def keep_cost(tr: ContextToolResult) -> int;
    def tiered_kind(tr: ContextToolResult, current_floor: int, kind: str) -> str;
    def smart_trim(
        tr: ContextToolResult, query_tokens: set[str], cg: ContextGraph
    ) -> bool;

    def micro_stub(tr: ContextToolResult, cg: ContextGraph, target: int = 0) -> bool;
    def apply_eviction(
        tr: ContextToolResult, kind: str, query_tokens: set[str], cg: ContextGraph
    ) -> str;

    def clear_to_budget(
        cg: ContextGraph, budget_tokens: int, present: Presence
    ) -> int;
}

def context_of(mt_run: object) -> Context | None;
def presence_of(mt_run: object) -> Presence;
def candidates_of(cg: ContextGraph, present: Presence) -> list[ContextToolResult];
def visible_tool_tokens(cg: ContextGraph, present: Presence) -> int;
def fill_tokens(cg: ContextGraph, present: Presence) -> int;
def current_turn_floor(cg: ContextGraph, present: Presence) -> int;
def scan_citations(cg: ContextGraph, cap: int = 20000);
def active_call_ids(cg: ContextGraph, turns: int, cap: int = 20000) -> dict[str, bool];
def focus_texts(cg: ContextGraph, present: Presence) -> dict[str, str];
def evicted_handle(tr: ContextToolResult) -> str;
def restore_if_trimmed(tr: ContextToolResult) -> bool;
def enforce_budget(
    ids: list[str], costs: dict[str, int], priority: dict[str, float], budget: int
) -> dict[str, bool];

def stub_reserve(n_candidates: int, budget: int, stub_chars: int) -> int;
def embed_text(model_name: str, text: str, clip: int) -> list[float];
def message_item_text(m: ContextMessage) -> str;
def bump_stat(cg: ContextGraph, key: str, n: int = 1);
