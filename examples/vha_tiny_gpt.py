# SPDX-License-Identifier: Apache-2.0

"""Train a tiny VHA GPT on synthetic tokens for a few single-GPU steps.

Run from the repository root after installing PaddleFleet and its GPU
dependencies:

    python examples/vha_tiny_gpt.py

This smoke case exercises VHA forward, backward, and optimizer updates. It is
not a quality benchmark or an inference-time weight-merging demonstration.
"""

import argparse

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

from paddlefleet import parallel_state
from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.models.gpt import GPTConfig


def _single_card_strategy():
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": 1,
        "sep_degree": 1,
        "cp_degree": 1,
        "ep_degree": 1,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    return strategy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if args.steps < 1 or args.seq_len < 2 or args.learning_rate <= 0:
        parser.error(
            "steps >= 1, seq-len >= 2, and learning-rate > 0 are required"
        )

    if (
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.device_count() == 0
    ):
        raise RuntimeError("This example requires one visible CUDA GPU.")
    paddle.set_device("gpu:0")
    paddle.manual_seed(42)

    strategy = _single_card_strategy()
    fleet.init(is_collective=True, strategy=strategy)
    if paddle.distributed.get_world_size() != 1:
        raise RuntimeError("Launch this example with exactly one process/GPU.")
    parallel_state.initialize_model_parallel(
        fleet.get_hybrid_communicate_group()
    )

    vocab_size = 128
    config = GPTConfig(
        num_hidden_layers=2,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        v_head_dim=32,
        vocab_size=vocab_size,
        max_sequence_length=args.seq_len,
        position_embedding_type="rope",
        normalization="RMSNorm",
        tie_word_embeddings=True,
        hidden_dropout_prob=0.0,
        attention_dropout=0.0,
        use_vha_attention=True,
        vha_q_lora_rank=16,
        vha_postmix_rank=2,
        multi_latent_attention=False,
        gated_attention=False,
        num_nextn_predict_layers=0,
        gpt_model_use_experimental_version=False,
    )
    model = gpt_builder(config, num_stages=1)
    model.train()
    pipeline = NoPipelineParallel(model, strategy)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=args.learning_rate,
        parameters=model.parameters(),
        weight_decay=0.0,
    )

    tokens = paddle.randint(1, vocab_size, [1, args.seq_len + 1])
    input_ids, labels = tokens[:, :-1], tokens[:, 1:]
    position_ids = paddle.arange(args.seq_len, dtype="int64").reshape(
        [1, args.seq_len]
    )
    vha_params = {
        name: param
        for name, param in model.named_parameters()
        if "vha_premix_weight" in name or "vha_postmix_V" in name
    }
    if not vha_params:
        raise RuntimeError(
            "No VHA parameters found; check the model configuration."
        )
    initial_values = {
        name: param.detach().clone() for name, param in vha_params.items()
    }

    for step in range(1, args.steps + 1):
        optimizer.clear_grad()
        batch = (
            {
                "input_ids": [input_ids],
                "position_ids": [position_ids],
            },
            [labels],
        )
        loss = pipeline.forward_backward_pipeline(batch)
        if not bool(paddle.isfinite(loss).item()):
            raise RuntimeError(f"Non-finite loss at step {step}.")
        for name, param in vha_params.items():
            if param.grad is None or not bool(
                paddle.all(paddle.isfinite(param.grad)).item()
            ):
                raise RuntimeError(f"Missing or non-finite gradient: {name}")
        optimizer.step()
        print(f"step={step} loss={float(loss.item()):.6f}")

    for parameter_kind in ("vha_premix_weight", "vha_postmix_V"):
        changed = any(
            bool(paddle.any(param.detach() != initial_values[name]).item())
            for name, param in vha_params.items()
            if parameter_kind in name
        )
        if not changed:
            raise RuntimeError(
                f"Optimizer step did not update {parameter_kind}."
            )
    print("VHA single-GPU optimizer smoke case passed.")


if __name__ == "__main__":
    main()
