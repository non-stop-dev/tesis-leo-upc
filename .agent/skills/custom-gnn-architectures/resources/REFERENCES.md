# Documentation Resources

This folder contains references to PyG documentation files that complement this skill.

## Primary References

- **04a_creating_message_passing_networks.md** - Complete tutorial on MessagePassing
- **17_torch_geometric_nn.md** - Full neural network layer API reference
- **31_gnn_cheatsheet.md** - Quick reference for GNN layers

## Key Concepts from Documentation

### The Message Passing Formula

From `04a_creating_message_passing_networks.md`:

$$
\mathbf{x}_i^{(k)} = \gamma^{(k)} \left( \mathbf{x}_i^{(k-1)}, \bigoplus_{j \in \mathcal{N}(i)} \phi^{(k)}\left(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)}, \mathbf{e}_{j,i}\right) \right)
$$

Where:
- $\phi$ = `message()` function
- $\bigoplus$ = `aggregate()` function  
- $\gamma$ = `update()` function

### MessagePassing Methods

| Method | Math Symbol | Purpose |
|--------|-------------|---------|
| `propagate()` | - | Start message passing |
| `message()` | $\phi$ | Construct messages |
| `aggregate()` | $\bigoplus$ | Combine messages |
| `update()` | $\gamma$ | Update embeddings |

### Variable Naming Convention

In `message()`, use suffixes to access node features:
- `x_j` → features of **source** (neighbor) nodes
- `x_i` → features of **target** (central) nodes

## GCN Layer Formula

$$
\mathbf{x}_i^{(k)} = \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{d_i} \cdot \sqrt{d_j}} \cdot \mathbf{W}^{\top} \cdot \mathbf{x}_j^{(k-1)} + \mathbf{b}
$$

## EdgeConv Formula

$$
\mathbf{x}_i^{(k)} = \max_{j \in \mathcal{N}(i)} h_\Theta(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)} - \mathbf{x}_i^{(k-1)})
$$
