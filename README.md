# Transformer_Toy

A C++ learning and experimentation framework for building, inspecting, and modifying Transformer-style architectures from first principles.

## Project Goal

The goal of this project is to build a small, readable Transformer implementation that can be used to explore architectural variants across multiple task domains, including text generation, code/debugging traces, graph-structured data, and possibly vision.

## Milestone 1 — Core Forward Logic

Status: Complete

Implemented:
- CPU Tensor class
- Random initialization utilities
- Linear layer
- Embedding layer
- LayerNorm
- Math utilities
  - softmax
  - ReLU
  - GELU
  - cross-entropy loss
  - perplexity
  - token accuracy
- Single-head causal self-attention
- FFN layer
- TransformerBlock
- Transformer model shell
- Character tokenizer
- Text dataset loader
- Random batch sampling
- Temperature sampling
- Top-k generation
- Basic evaluation metrics

## Milestone 2 — Multi-Head Attention and Architecture Controls

Status: Complete

Implemented:
- MultiHeadAttention layer
- Configurable attention type
  - single-head
  - multi-head
  - mixed per block
- Attention debug toggles
- ModelConfig and GenerationConfig structs
- Cleaner CLI/test harness
- Runtime controls for:
  - number of layers
  - embedding dimension
  - FFN hidden dimension
  - number of heads
  - context length
  - temperature
  - top-k

## Milestone 3 — Training Infrastructure

Status: Complete

Implemented::
- Manual backpropagation for selected layers
- Lightweight autograd engine
- Optimizer support
  - SGD
  - Adam
- Gradient checking utilities
- Training loop
- Checkpoint save/load
- Loss tracking over time

### Milestone 3.5 — CUDA Acceleration refactor
Status: In progress

Planned:
- GPU Dense/Linear
- GPU Transformer component pass
- Trainer device config
- CPU vs GPU parity tests

Completed:
- Refactor Tensor to be device-aware
- CUDA utility functions and error checking
- CPU/GPU copy and fill tests
- GPU SGD
- GPU Adam
- GPU CrossEntropyLoss

## Milestone 4 — Architectural Experiments

Planned:
- Deep FFN mixer
- Gated FFN / SwiGLU-style mixer
- CNN mixer
- GRU/LSTM mixer
- Hybrid block layouts
- Single-head vs multi-head comparisons
- Mixed attention-depth experiments

## Milestone 5 — Task Adapters

Planned:
- Text generation task
- Code/debug trace modeling
- Machine-code/debugging assistant experiments
- Graphormer-lite for graph tasks
- Possible image recognition adapter

## Milestone 6 — Research Evaluation

Planned:
- Compare architectures across task types
- Track loss, perplexity, accuracy, runtime, memory use
- Evaluate inductive bias suitability by domain
- Prepare results for possible paper/report

-------

## Design notes

- Input shape:        [batch, seq_len, d_model]
- Head count:         num_heads
- Head dimension:     d_head = d_model / num_heads
- Q/K/V projections:  d_model -> d_model
- Attention scores:   [batch, heads, seq_len, seq_len]
- Attention output:   [batch, seq_len, d_model]

_________

This project is a personal learning and experimentation tool, and is not intended for production use. It is designed to be simple and readable, rather than optimized for performance. Contributions and suggestions are welcome!