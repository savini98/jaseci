"""GraphMend TorchAttrTable: classify torch attributes as dynamic or static.

This is the generalization core of GraphMend's detection. Rather than hardcoding
a fixed set of ops (the prior implementation only looked at sum/max/min), we keep
a table of torch attributes/methods whose result is *data-dependent* -- i.e. it
returns a Python scalar, forces a CPU<->GPU sync, or produces a value/shape that
TorchDynamo cannot determine statically. When any such attribute appears in a
control-flow condition (directly, or transitively via a use-def chain), the
branch is treated as a data-dependent graph break per Algorithm 1 of the
GraphMend paper.

The table is intentionally easy to extend: add an attribute name to
`DYNAMIC_TORCH_ATTRS` to make GraphMend recognize a new dynamic op.
"""

# Torch ops/attributes whose result depends on tensor *data* (not just static
# shape), so using them in a branch condition forces a graph break.
glob DYNAMIC_TORCH_ATTRS: set[str] = {
         'item',
         'tolist',
         'sum',
         'max',
         'min',
         'mean',
         'prod',
         'argmax',
         'argmin',
         'nonzero',
         'count_nonzero',
         'all',
         'any',
         'equal',
         'allclose',
         'unique'
     },
     # Substrings (lower-cased) identifying logging-style side-effect calls.
     LOGGING_PATTERNS: list[str] = ['logger', 'logging', 'warn', 'critical'],
     # Built-in side-effecting calls that break tracing when seen inside a graph.
     SIDE_EFFECT_CALLS: set[str] = {'print'};

"""Return True if `attr` names a data-dependent (dynamic) torch attribute."""
def is_dynamic_torch_attr(attr: str) -> bool {
    return attr in DYNAMIC_TORCH_ATTRS;
}

"""Return True if `name` looks like a logging/printing side-effect call.

Matches the bare builtin `print` and any dotted/identifier call whose name
contains a logging keyword (e.g. `logger.warning`, `logging.info`).
"""
def is_side_effect_name(name: str) -> bool {
    if name in SIDE_EFFECT_CALLS {
        return True;
    }
    lowered = name.lower();
    for pattern in LOGGING_PATTERNS {
        if pattern in lowered {
            return True;
        }
    }
    return False;
}
