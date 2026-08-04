"""Registry of paper-evaluated models: how to build a small instance + inputs,
and which transformers submodule GraphMend should scope-transform.

Small configs (no weight download) are used so break-counting is fast; graph
breaks are structural (code paths), not weight-dependent. torch.manual_seed is
fixed by the runner so the two modes (graphmend on/off) get identical weights,
making the output fingerprints directly comparable for correctness.
"""
import torch


def _t5():
    from transformers import T5Config, T5ForConditionalGeneration
    cfg = T5Config(vocab_size=128, d_model=64, d_ff=128, num_layers=2,
                   num_heads=2, d_kv=32)
    m = T5ForConditionalGeneration(cfg)
    ids = torch.randint(0, 128, (1, 8))
    dec = torch.randint(0, 128, (1, 8))
    return m, {"input_ids": ids, "decoder_input_ids": dec}


def _biogpt():
    from transformers import BioGptConfig, BioGptForCausalLM
    cfg = BioGptConfig(vocab_size=128, hidden_size=64, num_hidden_layers=2,
                       num_attention_heads=2, intermediate_size=128,
                       max_position_embeddings=64)
    m = BioGptForCausalLM(cfg)
    return m, {"input_ids": torch.randint(0, 128, (1, 8))}


def _blenderbot():
    from transformers import BlenderbotConfig, BlenderbotForConditionalGeneration
    cfg = BlenderbotConfig(vocab_size=128, d_model=64, encoder_layers=2,
                           decoder_layers=2, encoder_attention_heads=2,
                           decoder_attention_heads=2, encoder_ffn_dim=128,
                           decoder_ffn_dim=128, max_position_embeddings=64)
    m = BlenderbotForConditionalGeneration(cfg)
    ids = torch.randint(0, 128, (1, 8))
    dec = torch.randint(0, 128, (1, 8))
    return m, {"input_ids": ids, "decoder_input_ids": dec}



def _marian():
    from transformers import MarianConfig, MarianMTModel
    cfg = MarianConfig(vocab_size=128, d_model=64, encoder_layers=2,
                       decoder_layers=2, encoder_attention_heads=2,
                       decoder_attention_heads=2, encoder_ffn_dim=128,
                       decoder_ffn_dim=128, max_position_embeddings=64,
                       decoder_start_token_id=2, pad_token_id=1)
    m = MarianMTModel(cfg)
    ids = torch.randint(3, 128, (1, 8))
    dec = torch.randint(3, 128, (1, 8))
    return m, {"input_ids": ids, "decoder_input_ids": dec}


def _pegasus_causal():
    from transformers import PegasusConfig, PegasusForCausalLM
    cfg = PegasusConfig(vocab_size=128, d_model=64, decoder_layers=2,
                        decoder_attention_heads=2, decoder_ffn_dim=128,
                        max_position_embeddings=64)
    m = PegasusForCausalLM(cfg)
    return m, {"input_ids": torch.randint(0, 128, (1, 8))}


def _phi3_longrope():
    """Phi-4-mini's LongRoPE path -- the paper's Figure 3 worked example.

    `rope_scaling.type = "longrope"` is what routes the forward through
    `longrope_frequency_update`, whose `seq_len > original_max_position_embeddings`
    test is the data-dependent branch [Where] rewrites. Without longrope the
    model takes the default rope path and exhibits no DC break at all.

    short/long_factor must each have length (hidden_size // num_attention_heads) // 2.
    """
    from transformers import Phi3Config, Phi3ForCausalLM
    hidden, heads = 64, 2
    n = (hidden // heads) // 2
    cfg = Phi3Config(
        vocab_size=128, hidden_size=hidden, intermediate_size=128, pad_token_id=0,
        num_hidden_layers=2, num_attention_heads=heads, num_key_value_heads=heads,
        max_position_embeddings=64, original_max_position_embeddings=16,
        rope_scaling={"type": "longrope",
                      "short_factor": [1.0] * n,
                      "long_factor": [2.0] * n},
    )
    m = Phi3ForCausalLM(cfg)
    return m, {"input_ids": torch.randint(0, 128, (1, 8))}


def _molformer():
    """MoLFormer-XL: the paper's [Trap] / validation-guard model (Table 2, VG 5).

    Its `MolformerSelfAttention.forward` holds the Figure 5 pattern verbatim --
    `if not torch.equal(attention_mask, ...): raise ValueError(...)`.

    Two things make this entry unlike the others. It is Hub REMOTE CODE, so it
    needs network access; and the revision is pinned, because the 2026-07
    "Fix deprecated code" commit retargeted the model at a newer transformers
    (it imports `transformers.masking_utils`, absent in the paper's pinned
    4.52.4). `7b12d946c181` is the last revision contemporary with the paper.
    """
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    repo, rev = "ibm/MoLFormer-XL-both-10pct", "7b12d946c181"
    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True, revision=rev)
    for k, v in [("hidden_size", 64), ("num_hidden_layers", 2),
                 ("num_attention_heads", 2), ("intermediate_size", 128),
                 ("max_position_embeddings", 64)]:
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cls = get_class_from_dynamic_module(
        "modeling_molformer.MolformerModel", repo, revision=rev)
    m = cls(cfg)
    return m, {"input_ids": torch.randint(0, 32, (1, 8)),
               "attention_mask": torch.ones(1, 8, dtype=torch.long)}


MODELS = {
    "t5-small":       {"build": _t5,         "scope": ["transformers.models.t5"]},
    "biogpt":         {"build": _biogpt,     "scope": ["transformers.models.biogpt"]},
    "blenderbot-400M-distill": {"build": _blenderbot, "scope": ["transformers.models.blenderbot"]},
    "opus-mt-fr-en":  {"build": _marian,        "scope": ["transformers.models.marian"]},
    "PegasusForCausalLM": {"build": _pegasus_causal, "scope": ["transformers.models.pegasus"]},
    # Phi-4-mini exercises [Where]. Its break site is `longrope_frequency_update`
    # in the SHARED top-level `transformers.modeling_rope_utils`, not under
    # `transformers.models.phi3` -- scoping only the model package silently
    # misses it and the model appears to be fixed by [Defer] alone.
    "Phi-4-mini-instruct": {"build": _phi3_longrope,
                            "scope": ["transformers.models.phi3",
                                      "transformers.modeling_rope_utils"]},
    # [Trap]. Opt-in: needs network + trust_remote_code, so it is excluded from
    # the default run (see NETWORK_MODELS). Hub remote code lands under the
    # `transformers_modules.*` namespace; scoping it works because jaclang hooks
    # the source loader as well as sys.meta_path (transformers builds the spec
    # directly, so a meta-path finder alone never sees these modules).
    "MoLFormer-XL-both10pct": {"build": _molformer,
                               "scope": ["transformers_modules"]},
}

# Models the default `python -m paper_eval.run_eval` skips: they download code or
# weights, which the rest of the harness deliberately avoids. Run them by name.
NETWORK_MODELS = {"MoLFormer-XL-both10pct"}
