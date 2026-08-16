"""The entry module for the decorator-factory region tests.

`forward` is a method of a proven torch Module subclass, so it seeds the
compiled region; its decorator is imported from region_deco_factory, so the
walk must cross the module boundary before it can enqueue the factory's
wrapper closure and, from there, the sibling closure the wrapper calls."""
import torch;
import from region_deco_factory { phase_wrap }

class PhaseNet(torch.nn.Module) {
    @phase_wrap
    def forward(self: any, x: any) -> any {
        return torch.relu(x);
    }
}
