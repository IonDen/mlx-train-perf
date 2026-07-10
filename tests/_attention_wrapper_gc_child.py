"""T12 subprocess child (run by test_attention_wrapper.py, never collected by pytest).

Proves the FULL production composition in ISOLATION from the parent pytest process
(repo gotcha 13: mlx_lm's grad_checkpoint patches `type(layer).__call__` at the CLASS
level and never reverts -- running a second gc=True train() in-process against the same
llama class nests checkpoints): a tiny (head_dim=64) llama whose every decoder layer's
attention is replaced by `enable_flash_attention` (impl="kernel", pre-warmed), then LoRA
fine-tuned for 2 real steps through mlx_lm's compiled `train()` with grad_checkpoint=True.
Success = the run completes with finite losses (a composition break -- the custom_function
vjp not firing under the class-level checkpoint patch inside mx.compile, or a graph error --
crashes or yields non-finite loss instead). Prints WRAPPER_GC_OK last.

This is the THIRD gc=True site; the other two are in-process against the same class and are
deliberately kept apart from it (gotcha 13): `tests/test_worker_train_step.py`
(`run_train_step` with grad_checkpoint) and `tests/_composition_gc_child.py` (T2's toy-vjp
composition child). Needs Metal (the kernel path); the parent test is `@pytest.mark.metal`.
"""
import math

import mlx.core as mx
import mlx.optimizers as optim
from mlx_lm.models import llama
from mlx_lm.tuner.trainer import default_loss
from mlx_lm.tuner.utils import linear_to_lora_layers

from mlx_train_perf.attention.wrapper import enable_flash_attention
from mlx_train_perf.bench.worker import _run_train_steps, _synthetic_train_examples
from mlx_train_perf.core.guards import install_guardrails

_SEQ_LEN = 16
_BATCH = 1
_STEPS = 2


def _tiny_llama_hd64() -> llama.Model:
    # head_dim = hidden_size // num_attention_heads = 128 // 2 = 64 (the kernel's smallest
    # supported head dim); GQA with num_key_value_heads=1 (group_size 2).
    args = llama.ModelArgs(
        model_type="llama", hidden_size=128, num_hidden_layers=2, intermediate_size=256,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=256, rms_norm_eps=1e-5,
        rope_theta=10000.0, tie_word_embeddings=False, head_dim=64,
    )
    return llama.Model(args)


def main() -> None:
    install_guardrails()
    model = _tiny_llama_hd64()
    enable_flash_attention(model, impl="kernel", seq_len=_SEQ_LEN, batch_size=_BATCH)

    mx.random.seed(0)  # BEFORE freeze/LoRA-injection: their random init draws next
    model.freeze()
    linear_to_lora_layers(model, -1, {"rank": 4, "dropout": 0.0, "scale": 20.0})
    mx.eval(model.parameters())

    examples = _synthetic_train_examples(
        vocab_size=256, seq_len=_SEQ_LEN, num_examples=_BATCH, seed=0
    )
    reports = _run_train_steps(
        model, optim.Adam(learning_rate=1e-4), default_loss, examples,
        batch=_BATCH, seq_len=_SEQ_LEN, steps=_STEPS, grad_checkpoint=True,
    )
    losses = [float(r["train_loss"]) for r in reports]
    assert len(losses) == _STEPS, losses
    assert all(math.isfinite(loss) for loss in losses), losses
    print(f"WRAPPER_GC_OK losses={losses}")


if __name__ == "__main__":
    main()
