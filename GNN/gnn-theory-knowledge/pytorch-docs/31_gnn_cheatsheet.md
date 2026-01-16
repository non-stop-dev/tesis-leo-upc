- `SparseTensor`: If checked (✓), supports message passing based on `torch_sparse.SparseTensor`, *e.g.*, `GCNConv(...).forward(x, adj_t)`. See [here](../advanced/sparse_tensor.html) for the accompanying tutorial.
- `edge_weight`: If checked (✓), supports message passing with one-dimensional edge weight information, *e.g.*, `GraphConv(...).forward(x, edge_index, edge_weight)`.
- `edge_attr`: If checked (✓), supports message passing with multi-dimensional edge feature information, *e.g.*, `GINEConv(...).forward(x, edge_index, edge_attr)`.
- **bipartite**: If checked (✓), supports message passing in bipartite graphs with potentially different feature dimensionalities for source and destination nodes, *e.g.*, `SAGEConv(in_channels=(16, 32), out_channels=64)`.
- **static**: If checked (✓), supports message passing in static graphs, *e.g.*, `GCNConv(...).forward(x, edge_index)` with `x` having shape `[batch_size, num_nodes, in_channels]`.
- **lazy**: If checked (✓), supports lazy initialization of message passing layers, *e.g.*, `SAGEConv(in_channels=-1, out_channels=64)`.


## Graph Neural Network Operators


| Name | SparseTensor | edge_weight | edge_attr | bipartite | static | lazy |
| --- | --- | --- | --- | --- | --- | --- |
| SimpleConv | ✓ | ✓ |  | ✓ | ✓ |  |
| GCNConv(Paper) | ✓ | ✓ |  |  | ✓ | ✓ |
| ChebConv(Paper) |  | ✓ |  |  | ✓ | ✓ |
| SAGEConv(Paper) | ✓ |  |  | ✓ | ✓ | ✓ |
| CuGraphSAGEConv(Paper) |  |  |  |  | ✓ |  |
| GraphConv(Paper) | ✓ | ✓ |  | ✓ | ✓ | ✓ |
| GatedGraphConv(Paper) | ✓ | ✓ |  |  | ✓ |  |
| ResGatedGraphConv(Paper) | ✓ |  | ✓ | ✓ | ✓ | ✓ |
| GATConv(Paper) | ✓ |  | ✓ | ✓ |  | ✓ |
| CuGraphGATConv(Paper) |  |  | ✓ |  | ✓ |  |
| FusedGATConv(Paper) |  |  |  |  | ✓ |  |
| GATv2Conv(Paper) | ✓ |  | ✓ | ✓ |  | ✓ |
| TransformerConv(Paper) | ✓ |  | ✓ | ✓ |  | ✓ |
| AGNNConv(Paper) | ✓ |  |  |  | ✓ |  |
| TAGConv(Paper) | ✓ | ✓ |  |  | ✓ | ✓ |
| GINConv(Paper) | ✓ |  |  | ✓ | ✓ |  |
| GINEConv(Paper) | ✓ |  | ✓ | ✓ | ✓ |  |
| ARMAConv(Paper) | ✓ | ✓ |  |  | ✓ | ✓ |
| SGConv(Paper) | ✓ | ✓ |  |  | ✓ | ✓ |
| SSGConv(Paper) | ✓ | ✓ |  |  | ✓ | ✓ |
| APPNP(Paper) | ✓ | ✓ |  |  | ✓ |  |
| MFConv(Paper) | ✓ |  |  | ✓ | ✓ | ✓ |
| DNAConv(Paper) | ✓ | ✓ |  |  |  |  |
| GMMConv(Paper) | ✓ |  | ✓ | ✓ | ✓ | ✓ |
| SplineConv(Paper) | ✓ |  | ✓ | ✓ | ✓ | ✓ |
| NNConv(Paper) | ✓ |  | ✓ | ✓ | ✓ | ✓ |
| CGConv(Paper) | ✓ |  | ✓ | ✓ | ✓ |  |
| EdgeConv(Paper) | ✓ |  |  | ✓ | ✓ |  |
| FeaStConv(Paper) | ✓ |  |  | ✓ | ✓ | ✓ |
| LEConv(Paper) | ✓ | ✓ |  | ✓ | ✓ | ✓ |
| PNAConv(Paper) | ✓ |  | ✓ |  |  | ✓ |
| ClusterGCNConv(Paper) | ✓ |  |  |  | ✓ | ✓ |
| GENConv(Paper) | ✓ |  | ✓ | ✓ | ✓ | ✓ |
| GCN2Conv(Paper) | ✓ | ✓ |  |  | ✓ |  |
| PANConv(Paper) | ✓ |  |  |  | ✓ | ✓ |
| WLConv(Paper) | ✓ |  |  |  | ✓ |  |
| WLConvContinuous(Paper) | ✓ | ✓ |  | ✓ | ✓ |  |
| SuperGATConv(Paper) | ✓ |  |  |  |  | ✓ |
| FAConv(Paper) | ✓ | ✓ |  | ✓ | ✓ | ✓ |
| EGConv(Paper) | ✓ |  |  |  |  | ✓ |
| PDNConv(Paper) | ✓ |  | ✓ |  | ✓ |  |
| GeneralConv(Paper) | ✓ |  | ✓ | ✓ |  | ✓ |
| LGConv(Paper) | ✓ | ✓ |  |  | ✓ |  |
| GPSConv(Paper) | ✓ |  |  |  | ✓ |  |
| AntiSymmetricConv(Paper) | ✓ |  |  |  | ✓ |  |
| DirGNNConv(Paper) |  |  |  |  | ✓ |  |
| MixHopConv(Paper) | ✓ | ✓ |  |  | ✓ | ✓ |
| MeshCNNConv(Paper) |  |  |  |  | ✓ |  |


## Heterogeneous Graph Neural Network Operators


| Name | SparseTensor | edge_weight | edge_attr | bipartite | static | lazy |
| --- | --- | --- | --- | --- | --- | --- |
| RGCNConv(Paper) | ✓ |  |  |  |  |  |
| FastRGCNConv | ✓ |  |  |  |  |  |
| CuGraphRGCNConv(Paper) |  |  |  |  | ✓ |  |
| RGATConv(Paper) | ✓ |  | ✓ |  |  |  |
| FiLMConv(Paper) | ✓ |  |  | ✓ | ✓ | ✓ |
| HGTConv(Paper) | ✓ |  |  |  |  | ✓ |
| HEATConv(Paper) | ✓ |  | ✓ |  |  | ✓ |
| HeteroConv |  |  |  |  | ✓ |  |
| HANConv(Paper) | ✓ |  |  |  |  | ✓ |


## Hypergraph Neural Network Operators


| Name | SparseTensor | edge_weight | edge_attr | bipartite | static | lazy |
| --- | --- | --- | --- | --- | --- | --- |
| HypergraphConv(Paper) |  | ✓ | ✓ |  |  | ✓ |


## Point Cloud Neural Network Operators


| Name | bipartite | lazy |
| --- | --- | --- |
| GravNetConv(Paper) | ✓ | ✓ |
| SignedConv(Paper) | ✓ | ✓ |
| PointNetConv(Paper) | ✓ |  |
| DynamicEdgeConv(Paper) | ✓ |  |
| XConv(Paper) |  |  |
| PPFConv(Paper) | ✓ |  |
| PointTransformerConv(Paper) | ✓ | ✓ |
| HeteroConv |  |  |
| PointGNNConv(Paper) |  |  |


