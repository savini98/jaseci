"""A decorator factory whose wrapper closure calls a sibling closure.

`phase_wrap` returns `wrapper`, which replaces the decorated forward at class
definition time. `wrapper` calls `phase_update`, a sibling closure bound only
in the factory's local scope, so the region walk must resolve the bare name
through the enclosing function scope to reach the data-dependent branch and
the print inside it. `phase_orphan` is a sibling the wrapper never calls: it
must stay outside the region and compile byte-identical."""
import torch;

def phase_wrap(inner_forward: any) -> any {
    def phase_update(self: any, x: any) -> any {
        if x.sum() > 0 {
            y = x * 2.0;
        } else {
            y = x * 3.0;
        }
        print("PHASE-UPDATE", y);
        return y;
    }
    def phase_orphan(self: any, x: any) -> any {
        if x.sum() > 0 {
            y = x * 4.0;
        } else {
            y = x * 5.0;
        }
        print("PHASE-ORPHAN", y);
        return y;
    }
    def wrapper(self: any, x: any) -> any {
        y = phase_update(self, x);
        return inner_forward(self, y);
    }
    return wrapper;
}
