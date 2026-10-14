# Virtual Head Attention (VHA)

Xibo Ding*, Guoxia Wang*, Jinle Zeng, Jiabin Yang, Sirui Tao, Chao Yang, Dianhai Yu

*NeurIPS 2026 · [📄 Paper](https://arxiv.org/abs/2610.17959)*

**Train with virtual heads. Infer with standard GQA.** Fewer KV heads reduce the cache of grouped-query attention (GQA); fewer physical query heads reduce training and prefill compute. Reducing both can cost model quality.

Virtual Head Attention (VHA) addresses that loss with two training-time operations: **Q Premix** expands fewer physical query heads into virtual heads within each KV group, while **Linear Postmix** mixes attention outputs across groups. In the paper's linear formulation, both operations fold into the query and output projections after training, leaving standard GQA-2 inference without a VHA-specific attention kernel.

This repository contains the PaddleFleet training implementation, a runnable single-GPU GPT example, and tests for the VHA attention path.

![VHA overview: Q Premix and Linear Postmix fold into standard GQA at inference](assets/vha_overview.png)

*Q Premix (orange) creates virtual queries before attention; Linear Postmix (purple) mixes the resulting head outputs. The dashed paths show where their weights are absorbed for inference.*

## Results

VHA improves the quality–cache–compute trade-off across four matched pretraining scales. The 1.7B control shows what its training-time reparameterization adds beyond simply using fewer heads. All three models below were pretrained on 50B tokens at 4K context; accuracy is the zero-shot average over eight tasks.

| Attention (physical Q-K-V heads) | Eight-task average ↑ | KV cache ↓ |
| --- | ---: | ---: |
| GQA (16-8-8) | 54.09 | 4.00 KB |
| GQA (8-2-2) | 52.38 | 1.00 KB |
| VHA (8-2-2) | **54.50** | 1.00 KB |

*KV-cache values are per layer per token in BF16; VHA expands its eight physical query heads into sixteen virtual heads.* Simply compressing GQA to the same 8-2-2 head configuration loses 1.71 points; VHA recovers that loss and exceeds the original GQA-8 baseline by 0.41 points. Its calculated training/prefill FLOPs are also 7.9% lower than GQA-8's at 4K context.

The full four-scale study covers three dense models from 0.6B to 4B and an 8B-A1B MoE, each trained for 50B tokens at 4K context. VHA has the highest eight-task average among the compared methods at every scale. Against the corresponding GQA baselines, it uses a 2–4× smaller per-layer KV cache and 6.5–8.1% fewer calculated training/prefill FLOPs.

Beyond these 50B-token runs, the paper tests scale and context separately:

- **Larger model:** At 30B-A3B, after 500B training tokens at 8K context, VHA scores 50.36 on the four-category macro-average of 12 base-model benchmarks, compared with 49.66 for MLA and 47.76 for GQA-4. Its BF16 KV cache per layer per token is 1.00 KB, compared with 1.12 KB and 2.00 KB, respectively.
- **Long context:** In a separate matched 10B-A1B hybrid-attention study, VHA leads GQA on RULER at every evaluated length from 8K to 128K. At the final 128K checkpoint, its average across lengths is 84.4 versus 73.0; the two models differ in their full-attention layers, which use two versus four KV heads.

See the [paper](https://arxiv.org/html/2610.17959v1) for full benchmark results and experimental protocols.

## 🚀 Quick start

On Linux x86-64 with an NVIDIA GPU and a CUDA 13-compatible driver, run the example with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync --locked --python 3.12
CUDA_VISIBLE_DEVICES=0 uv run --locked python examples/vha_tiny_gpt.py
```

The [example](examples/vha_tiny_gpt.py) runs two FP32 optimizer steps on synthetic tokens with a two-layer GPT. It checks finite losses, VHA gradients, and parameter updates as a quick integration smoke test.

## Inside the code

- **Configure:** Set `use_vha_attention=True` in [`GPTConfig`](src/paddlefleet/models/gpt/gpt_config.py). The [transformer configuration](src/paddlefleet/transformer/transformer_config.py) defines VHA's Premix and Postmix options; the [example](examples/vha_tiny_gpt.py) shows them in a complete GPT setup.
- **Trace:** [GPT layer selection](src/paddlefleet/models/gpt/gpt_layer_specs.py) routes to [`SelfAttentionVHA`](src/paddlefleet/transformer/attention.py), which implements the VHA attention path.
- **Check:** The [single-card tests](tests/single_card_tests/test_vha_attention.py) exercise that path.

The current VHA implementation requires tensor-parallel degree 1.

## Citation

If you use VHA, please cite:

```bibtex
@inproceedings{ding2026virtualheadattention,
  title={{Virtual Head Attention}},
  author={Ding, Xibo and Wang, Guoxia and Zeng, Jinle and Yang, Jiabin and Tao, Sirui and Yang, Chao and Yu, Dianhai},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  eprint={2610.17959},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2610.17959}
}
```

## License

This code builds on PaddleFleet and is licensed under [Apache 2.0](LICENSE).
