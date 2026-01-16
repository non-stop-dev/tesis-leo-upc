# References

## Academic Papers

### Semi-Supervised Classification with Graph Convolutional Networks (GCN)
**Citation**: Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*.

**Key Contributions**:
- Introduced the spectral rule $H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$.
- Bridged spectral graph theory with spatial message passing.
- Proven effective for MSME survival prediction by capturing local neighborhood structures.

### Graph Attention Networks (GAT)
**Citation**: Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph attention networks. *ICLR*.

**Key Contributions**:
- Introduced attention mechanisms to weigh neighbor importance.
- Allows the model to focus on relevant competitors or supply chain partners.

### Distill: Understanding Graph Convolutions
**Citation**: Daigavane, A., Ravindran, B., & Aggarwal, G. (2021). Understanding Convolutions on Graphs. *Distill*.

**Key Concepts**:
- **Polynomial Filters**: Expressing convolutions as polynomials of the Laplacian $L$.
- **Chebyshev Polynomials (ChebNet)**: Fast approximation of spectral filters avoiding eigen-decomposition.

## Documentation
- [PyG MessagePassing](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.MessagePassing)
