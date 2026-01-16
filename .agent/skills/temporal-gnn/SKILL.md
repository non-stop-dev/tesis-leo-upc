---
name: temporal-gnn
description: Handling graphs that change over time (nodes/edges added or attributes evolving). Covers Discrete (snapshot-based) and Continuous (TGN) approaches. Use when your dataset has a time dimension (e.g., survival of MSMEs over multiple years).
---

# Temporal GNN Skill

This skill guides the modeling of **Dynamic Graphs** where structure or features evolve over time.

## When to use this skill

- Your graph changes over time (not just a static snapshot).
- You have timestamps for edges (e.g., transaction time, interaction time).
- You need to predict future links or node states based on history.

## Types of Dynamic Graphs

### 1. Discrete Dynamic Graphs (Snapshots)
Graph is observed at fixed intervals $t=1, \dots, T$.
- **Representation**: $\mathcal{G} = \{G_1, G_2, \dots, G_T\}$
- **Approaches**:
    - **Stacked DGNN**: Apply GNN on each $G_t$, then feed embeddings to RNN/LSTM.
    - **Integrated DGNN**: GNN weights evolve via RNN (e.g., **EvolveGCN**).
- **Use Case**: Census data available every year (2020, 2021, 2022).

### 2. Continuous Dynamic Graphs (Streaming)
Graph changes continuously; edges are events $(u, v, t)$.
- **Representation**: Stream of events.
- **Approaches**:
    - **TGN (Temporal Graph Networks)**: Maintains a "memory" state for each node, updated by events.
    - **JODIE / TGAT**: Similar continuous time approaches.
- **Use Case**: Real-time transaction logs, social media interactions.

## Key Architectures (from Survey)

### EvolveGCN (Discrete)
- Adapts the GCN weights $\mathbf{W}_t$ at each step using an RNN.
- $\mathbf{W}_t = \text{GRU}(\mathbf{W}_{t-1}, ...)$
- Good for handling new nodes appearing in snapshots.

### TGN (Continuous)
- **Memory**: $s_i(t)$ state for each node.
- **Message**: Interaction $(u, v, t)$ creates messages for $u$ and $v$.
- **Update**: Memory updated via GRU: $s_i(t) = \text{GRU}(s_i(t^-), \text{msg})$.
- **Embedding**: $z_i(t) = \text{GNN}(s_i(t), s_j(t), \dots)$ from temporal neighbors.

## Decision Tree

```
Is your data a stream of events or snapshots?
├── Snapshots (e.g. yearly census) → Discrete DGNN
│   ├── Need to handle new nodes? → EvolveGCN
│   └── Fixed nodes? → GCRN / LRCN
└── Event Stream (e.g. transactions) → Continuous DGNN (TGN)
```

## References
- `dynamic_gnn_survey.md` - Comprehensive survey.
- TGN Paper: *Temporal Graph Networks for Deep Learning on Dynamic Graphs* (Rossi et al., 2020)
- EvolveGCN Paper: *EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs* (Pareja et al., 2020)
