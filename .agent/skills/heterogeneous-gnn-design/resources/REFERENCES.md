# References

## Academic Papers

### Heterogeneous Graph Attention Network (HAN)
**Citation**: Wang, X., Ji, H., Shi, C., Wang, B., Ye, Y., Cui, P., & Yu, P. S. (2019). Heterogeneous graph attention network. *WWW*.

**Key Contributions**:
- **Meta-path Attention**: Aggregates information along semantic paths (e.g., Paper-Author-Paper).
- **Semantic Attention**: Learns the importance of different meta-paths.

### MAGNN: Metapath Aggregated Graph Neural Network
**Citation**: Fu, X., Zhang, J., Meng, Z., & King, I. (2020). Magnn: Metapath aggregated graph neural network for heterogeneous graph embedding. *WWW*.

**Key Contributions**:
- Considers intermediate nodes in meta-paths, not just end-points.
- Reduces information loss compared to HAN.

### Graph Transformer Networks (GTN)
**Citation**: Yun, S., Jeong, M., Kim, R., Kang, J., & Kim, H. J. (2019). Graph transformer networks. *NeurIPS*.

**Key Contributions**:
- Learns new graph structures (soft selection of meta-paths) via transformer mechanisms.
- Useful when useful meta-paths are unknown.

## Documentation
- [PyTorch Geometric Heterogeneous Graph Learning](https://pytorch-geometric.readthedocs.io/en/latest/tutorial/heterogeneous.html)
