"""T2 subprocess child (run by test_attention_composition.py, never collected by pytest).

Proves the full production composition in ISOLATION from the parent pytest process
(gotcha 13: mlx_lm's grad_checkpoint patches type(layer).__call__ and never reverts):
a tiny llama whose EVERY attention output routes through an mx.custom_function with a
hand-written vjp, trained for 2 real steps through mlx_lm's compiled train() with
grad_checkpoint=True. Success = the run completes with finite losses (a composition
break — vjp not invoked under the class-level checkpoint patch inside mx.compile, or a
graph error — crashes or yields non-finite loss instead). Prints COMPOSITION_OK last.
"""
import math

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx_lm.models import llama
from mlx_lm.tuner.trainer import default_loss

from mlx_train_perf.bench.worker import _run_train_steps, _synthetic_train_examples
from mlx_train_perf.core.guards import install_guardrails

_SENTINEL = 3.0


@mx.custom_function
def sentinel_identity(x: mx.array) -> mx.array:
    return x * 1.0


@sentinel_identity.vjp
def _sentinel_identity_vjp(
    _primals: mx.array, cotangent: mx.array, _outputs: mx.array
) -> mx.array:
    # Value-identity forward; gradient scaled by _SENTINEL -- if this vjp is NOT invoked,
    # training still runs (autodiff through x*1.0), so the parent's mx-level sentinel
    # tests carry the engagement proof; THIS script proves the full-stack composition
    # (class-level checkpoint patch x mx.compile x custom_function) runs end to end.
    return cotangent * _SENTINEL


class SentinelAttention(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:  # type: ignore[no-untyped-def]
        return sentinel_identity(self.inner(x, mask, cache))


def main() -> None:
    install_guardrails()
    args = llama.ModelArgs(
        model_type="llama", hidden_size=64, num_hidden_layers=2, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=256, rms_norm_eps=1e-5,
        rope_theta=10000.0, tie_word_embeddings=False,
    )
    model = llama.Model(args)
    for layer in model.model.layers:
        layer.self_attn = SentinelAttention(layer.self_attn)

    examples = _synthetic_train_examples(
        vocab_size=256, seq_len=16, num_examples=1, seed=0
    )
    reports = _run_train_steps(
        model, optim.Adam(learning_rate=1e-5), default_loss, examples,
        batch=1, seq_len=16, steps=2, grad_checkpoint=True,
    )
    losses = [float(r["train_loss"]) for r in reports]  # type: ignore[arg-type]
    assert len(losses) == 2, losses
    assert all(math.isfinite(loss) for loss in losses), losses
    print(f"COMPOSITION_OK losses={losses}")


if __name__ == "__main__":
    main()
