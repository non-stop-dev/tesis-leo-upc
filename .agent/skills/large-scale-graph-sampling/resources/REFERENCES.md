# References

## Academic Papers

### GraphSAGE
**Citation**: Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. *Advances in neural information processing systems*, 30.

**Key Contributions**:
- **Inductive Learning**: generalizing to unseen nodes (vital for MSME survival prediction on new companies).
- **Neighbor Sampling**: Fixed-size sampling to handle "neighbor explosion".
- **Aggregators**:
    - **Mean**: Element-wise mean of vectors (simple, efficient).
    - **LSTM**: Powerful sequential aggregation (requires random permutation).
    - **Pooling**: Element-wise max/mean pooling after MLP transformation.

**Key Formulas**:

*Forward Propagation (Algorithm 1)*:
$$
\mathbf{h}^k_{\mathcal{N}(v)} \leftarrow \text{AGGREGATE}_k(\{\mathbf{h}^{k-1}_u, \forall u \in \mathcal{N}(v)\})
$$
$$
\mathbf{h}^k_v \leftarrow \sigma(\mathbf{W}^k \cdot \text{CONCAT}(\mathbf{h}^{k-1}_v, \mathbf{h}^k_{\mathcal{N}(v)}))
$$

*Loss Function (Unsupervised)*:
$$
J_{\mathcal{G}}(\mathbf{z}_u) = -\log(\sigma(\mathbf{z}_u^\top \mathbf{z}_v)) - Q \cdot \mathbb{E}_{v_n \sim P_n(v)} \log(\sigma(-\mathbf{z}_u^\top \mathbf{z}_{v_n}))
$$

**relevance to MSME**:
This is the foundational paper for the `NeighborLoader` used in the PyG implementation. It proves that sampling neighbors doesn't just save memory—it regularizes training and allows for inductive inference on the 1.3M MSME dataset.

## Documentation
- [PyTorch Geometric NeighborLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.NeighborLoader)
