> **Warning:** `torch_geometric.distributed` has been deprecated since 2.7.0 and will
no longer be maintained. For distributed training, refer to [our
tutorials on distributed training](../tutorial/distributed.html#distributed-tutorials) or [cuGraph
examples](https://github.com/rapidsai/cugraph-gnn/tree/main/python/cugraph-pyg/cugraph_pyg/examples).


| DistContext | Context information of the current process. |
| --- | --- |
| LocalFeatureStore | Implements theFeatureStoreinterface to act as a local feature store for distributed training. |
| LocalGraphStore | Implements theGraphStoreinterface to act as a local graph store for distributed training. |
| Partitioner | Partitions the graph and its features of aDataorHeteroDataobject. |
| DistNeighborSampler | An implementation of a distributed and asynchronised neighbor sampler used byDistNeighborLoaderandDistLinkNeighborLoader. |
| DistLoader | A base class for creating distributed data loading routines. |
| DistNeighborLoader | A distributed loader that performs sampling from nodes. |
| DistLinkNeighborLoader | A distributed loader that performs sampling from edges. |


***class *DistContext(*rank: [int](https://docs.python.org/3/library/functions.html#int)*, *global_rank: [int](https://docs.python.org/3/library/functions.html#int)*, *world_size: [int](https://docs.python.org/3/library/functions.html#int)*, *global_world_size: [int](https://docs.python.org/3/library/functions.html#int)*, *group_name: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *role: DistRole = DistRole.WORKER*)[[source]](../_modules/torch_geometric/distributed/dist_context.html#DistContext)**
: Context information of the current process.


***class *LocalFeatureStore[[source]](../_modules/torch_geometric/distributed/local_feature_store.html#LocalFeatureStore)**
: Implements the [FeatureStore](../generated/torch_geometric.data.FeatureStore.html#torch_geometric.data.FeatureStore) interface to
act as a local feature store for distributed training.


**get_all_tensor_attrs() → [List](https://docs.python.org/3/library/typing.html#typing.List)[LocalTensorAttr][[source]](../_modules/torch_geometric/distributed/local_feature_store.html#LocalFeatureStore.get_all_tensor_attrs)**
: Returns all registered tensor attributes.


**Return type:**
: [List](https://docs.python.org/3/library/typing.html#typing.List)[`LocalTensorAttr`]


**lookup_features(*index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *is_node_feat: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, *input_type: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]]] = None*) → [Future](https://docs.pytorch.org/docs/main/futures.html#torch.futures.Future)[[source]](../_modules/torch_geometric/distributed/local_feature_store.html#LocalFeatureStore.lookup_features)**
: Lookup of local/remote features.


**Return type:**
: [Future](https://docs.pytorch.org/docs/main/futures.html#torch.futures.Future)


***classmethod *from_data(*node_id: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *x: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *y: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *edge_id: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*) → LocalFeatureStore[[source]](../_modules/torch_geometric/distributed/local_feature_store.html#LocalFeatureStore.from_data)**
: Creates a local feature store from homogeneous PyG tensors.


**Parameters:**
: - **node_id** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The global identifier for every local node.
- **x** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The node features.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **y** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The node labels. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **edge_id** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The global identifier for every
local edge. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **edge_attr** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The edge features.
(default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: LocalFeatureStore


***classmethod *from_hetero_data(*node_id_dict: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *x_dict: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]] = None*, *y_dict: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]] = None*, *edge_id_dict: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]] = None*, *edge_attr_dict: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]] = None*) → LocalFeatureStore[[source]](../_modules/torch_geometric/distributed/local_feature_store.html#LocalFeatureStore.from_hetero_data)**
: Creates a local graph store from heterogeneous PyG tensors.


**Parameters:**
: - **node_id_dict** (*Dict**[**NodeType**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]*) – The global identifier
for every local node of every node type.
- **x_dict** (*Dict**[**NodeType**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]**, **optional*) – The node features
of every node type. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **y_dict** (*Dict**[**NodeType**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]**, **optional*) – The node labels of
every node type. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **edge_id_dict** (*Dict**[**EdgeType**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]**, **optional*) – The global
identifier for every local edge of every edge types.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **edge_attr_dict** (*Dict**[**EdgeType**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]**, **optional*) – The edge
features of every edge type. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: LocalFeatureStore


***class *LocalGraphStore[[source]](../_modules/torch_geometric/distributed/local_graph_store.html#LocalGraphStore)**
: Implements the [GraphStore](../generated/torch_geometric.data.GraphStore.html#torch_geometric.data.GraphStore) interface to
act as a local graph store for distributed training.


**get_partition_ids_from_nids(*ids: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *node_type: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/distributed/local_graph_store.html#LocalGraphStore.get_partition_ids_from_nids)**
: Returns the partition IDs of node IDs for a specific node type.


**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


**get_partition_ids_from_eids(*eids: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_type: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*)[[source]](../_modules/torch_geometric/distributed/local_graph_store.html#LocalGraphStore.get_partition_ids_from_eids)**
: Returns the partition IDs of edge IDs for a specific edge type.


**get_all_edge_attrs() → [List](https://docs.python.org/3/library/typing.html#typing.List)[[EdgeAttr](../generated/torch_geometric.data.EdgeAttr.html#torch_geometric.data.EdgeAttr)][[source]](../_modules/torch_geometric/distributed/local_graph_store.html#LocalGraphStore.get_all_edge_attrs)**
: Returns all registered edge attributes.


**Return type:**
: [List](https://docs.python.org/3/library/typing.html#typing.List)[[EdgeAttr](../generated/torch_geometric.data.EdgeAttr.html#torch_geometric.data.EdgeAttr)]


***classmethod *from_data(*edge_id: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *num_nodes: [int](https://docs.python.org/3/library/functions.html#int)*, *is_sorted: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → LocalGraphStore[[source]](../_modules/torch_geometric/distributed/local_graph_store.html#LocalGraphStore.from_data)**
: Creates a local graph store from a homogeneous or heterogenous
PyG graph.


**Parameters:**
: - **edge_id** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The global identifier for every local edge.
- **edge_index** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The local edge indices.
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)) – The number of nodes in the local graph.
- **is_sorted** ([bool](https://docs.python.org/3/library/functions.html#bool)) – Whether edges are sorted by column/destination
nodes (CSC format). (default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: LocalGraphStore


***classmethod *from_hetero_data(*edge_id_dict: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *edge_index_dict: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *num_nodes_dict: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)]*, *is_sorted: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → LocalGraphStore[[source]](../_modules/torch_geometric/distributed/local_graph_store.html#LocalGraphStore.from_hetero_data)**
: Creates a local graph store from a heterogeneous PyG graph.


**Parameters:**
: - **edge_id_dict** (*Dict**[**EdgeType**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]*) – The global identifier
for every local edge of every edge type.
- **edge_index_dict** (*Dict**[**EdgeType**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]*) – The local edge
indices of every edge type.
- **num_nodes_dict** ([Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)]) – (Dict[str, int]): The number of nodes for every
node type.
- **is_sorted** ([bool](https://docs.python.org/3/library/functions.html#bool)) – Whether edges are sorted by column/destination
nodes (CSC format). (default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: LocalGraphStore


***class *Partitioner(*data: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)]*, *num_parts: [int](https://docs.python.org/3/library/functions.html#int)*, *root: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *recursive: [bool](https://docs.python.org/3/library/functions.html#bool) = False*)[[source]](../_modules/torch_geometric/distributed/partition.html#Partitioner)**
: Partitions the graph and its features of a
[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) or
[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData) object.


Partitioned data output will be structured as shown below.


**Homogeneous graphs:**


```
root/
|-- META.json
|-- node_map.pt
|-- edge_map.pt
|-- part0/
    |-- graph.pt
    |-- node_feats.pt
    |-- edge_feats.pt
|-- part1/
    |-- graph.pt
    |-- node_feats.pt
    |-- edge_feats.pt
```


**Heterogeneous graphs:**


```
root/
|-- META.json
|-- node_map/
    |-- ntype1.pt
    |-- ntype2.pt
|-- edge_map/
    |-- etype1.pt
    |-- etype2.pt
|-- part0/
    |-- graph.pt
    |-- node_feats.pt
    |-- edge_feats.pt
|-- part1/
    |-- graph.pt
    |-- node_feats.pt
    |-- edge_feats.pt
```


**Parameters:**
: - **data** ([Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)* or *[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)) – The data object.
- **num_parts** ([int](https://docs.python.org/3/library/functions.html#int)) – The number of partitions.
- **recursive** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will use multilevel
recursive bisection instead of multilevel k-way partitioning.
(default: [False](https://docs.python.org/3/library/constants.html#False))
- **root** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – Root directory where the partitioned dataset should be
saved.


**generate_partition()[[source]](../_modules/torch_geometric/distributed/partition.html#Partitioner.generate_partition)**
: Generates the partitions.


***class *DistNeighborSampler(*current_ctx: DistContext*, *data: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[LocalFeatureStore, LocalGraphStore]*, *num_neighbors: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[NumNeighbors](sampler.html#torch_geometric.sampler.NumNeighbors), [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]]*, *channel: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[Queue] = None*, *replace: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *subgraph_type: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[SubgraphType, [str](https://docs.python.org/3/library/stdtypes.html#str)] = 'directional'*, *disjoint: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *temporal_strategy: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'uniform'*, *time_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *concurrency: [int](https://docs.python.org/3/library/functions.html#int) = 1*, *device: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[device](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.device)] = None*, ***kwargs*)[[source]](../_modules/torch_geometric/distributed/dist_neighbor_sampler.html#DistNeighborSampler)**
: An implementation of a distributed and asynchronised neighbor sampler
used by DistNeighborLoader and
DistLinkNeighborLoader.


***async *node_sample(*inputs: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[NodeSamplerInput](sampler.html#torch_geometric.sampler.NodeSamplerInput), DistEdgeHeteroSamplerInput]*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[SamplerOutput](sampler.html#torch_geometric.sampler.SamplerOutput), [HeteroSamplerOutput](sampler.html#torch_geometric.sampler.HeteroSamplerOutput)][[source]](../_modules/torch_geometric/distributed/dist_neighbor_sampler.html#DistNeighborSampler.node_sample)**
: Performs layer-by-layer distributed sampling from a
`NodeSamplerInput` or `DistEdgeHeteroSamplerInput` and
returns the output of the sampling procedure.


> **Note:** In case of distributed training it is required to synchronize the
results between machines after each layer.


**Return type:**
: `Union`[[SamplerOutput](sampler.html#torch_geometric.sampler.SamplerOutput), [HeteroSamplerOutput](sampler.html#torch_geometric.sampler.HeteroSamplerOutput)]


***async *edge_sample(*inputs: [EdgeSamplerInput](sampler.html#torch_geometric.sampler.EdgeSamplerInput)*, *sample_fn: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*, *num_nodes: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [int](https://docs.python.org/3/library/functions.html#int)]]*, *disjoint: [bool](https://docs.python.org/3/library/functions.html#bool)*, *node_time: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]] = None*, *neg_sampling: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[NegativeSampling](sampler.html#torch_geometric.sampler.NegativeSampling)] = None*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[SamplerOutput](sampler.html#torch_geometric.sampler.SamplerOutput), [HeteroSamplerOutput](sampler.html#torch_geometric.sampler.HeteroSamplerOutput)][[source]](../_modules/torch_geometric/distributed/dist_neighbor_sampler.html#DistNeighborSampler.edge_sample)**
: Performs layer-by-layer distributed sampling from an
`EdgeSamplerInput` and returns the output of the sampling
procedure.


> **Note:** In case of distributed training it is required to synchronize the
results between machines after each layer.


**Return type:**
: `Union`[[SamplerOutput](sampler.html#torch_geometric.sampler.SamplerOutput), [HeteroSamplerOutput](sampler.html#torch_geometric.sampler.HeteroSamplerOutput)]


***async *sample_one_hop(*srcs: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *one_hop_num: [int](https://docs.python.org/3/library/functions.html#int)*, *seed_time: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *src_batch: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *edge_type: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*) → [SamplerOutput](sampler.html#torch_geometric.sampler.SamplerOutput)[[source]](../_modules/torch_geometric/distributed/dist_neighbor_sampler.html#DistNeighborSampler.sample_one_hop)**
: Samples one-hop neighbors for a set of seed nodes in `srcs`.
If seed nodes are located on a local partition, evaluates the sampling
function on the current machine. If seed nodes are from a remote
partition, sends a request to a remote machine that contains this
partition.


**Return type:**
: [SamplerOutput](sampler.html#torch_geometric.sampler.SamplerOutput)


***class *DistLoader(*current_ctx: DistContext*, *master_addr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *master_port: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*, *channel: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[Queue] = None*, *num_rpc_threads: [int](https://docs.python.org/3/library/functions.html#int) = 16*, *rpc_timeout: [int](https://docs.python.org/3/library/functions.html#int) = 180*, *dist_sampler: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[DistNeighborSampler] = None*, ***kwargs*)[[source]](../_modules/torch_geometric/distributed/dist_loader.html#DistLoader)**
: A base class for creating distributed data loading routines.


**Parameters:**
: - **current_ctx** (DistContext) – Distributed context info of the current
process.
- **master_addr** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – RPC address for distributed loader
communication.
Refers to the IP address of the master node. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **master_port** ([int](https://docs.python.org/3/library/functions.html#int)* or *[str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The open port for RPC communication
with the master node. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **channel** (*mp.Queue**, **optional*) – A communication channel for messages.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_rpc_threads** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of threads in the
thread-pool used by
`TensorPipeAgent` to execute
requests. (default: `16`)
- **rpc_timeout** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The default timeout in seconds for RPC
requests.
If the RPC has not completed in this timeframe, an exception will
be raised.
Callers can override this timeout for
individual RPCs in `rpc_sync()` and
`rpc_async()` if necessary.
(default: `180`)


***class *DistNeighborLoader(*data: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[LocalFeatureStore, LocalGraphStore]*, *num_neighbors: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]]*, *master_addr: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *master_port: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [str](https://docs.python.org/3/library/stdtypes.html#str)]*, *current_ctx: DistContext*, *input_nodes: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [None](https://docs.python.org/3/library/constants.html#None), [str](https://docs.python.org/3/library/stdtypes.html#str), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]] = None*, *input_time: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *dist_sampler: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[DistNeighborSampler] = None*, *replace: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *subgraph_type: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[SubgraphType, [str](https://docs.python.org/3/library/stdtypes.html#str)] = 'directional'*, *disjoint: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *temporal_strategy: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'uniform'*, *time_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *transform: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)] = None*, *concurrency: [int](https://docs.python.org/3/library/functions.html#int) = 1*, *num_rpc_threads: [int](https://docs.python.org/3/library/functions.html#int) = 16*, *filter_per_worker: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[bool](https://docs.python.org/3/library/functions.html#bool)] = False*, *async_sampling: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, *device: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[device](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.device)] = None*, ***kwargs*)[[source]](../_modules/torch_geometric/distributed/dist_neighbor_loader.html#DistNeighborLoader)**
: A distributed loader that performs sampling from nodes.


**Parameters:**
: - **data** ([tuple](https://docs.python.org/3/library/stdtypes.html#tuple)) – A ([FeatureStore](../generated/torch_geometric.data.FeatureStore.html#torch_geometric.data.FeatureStore),
[GraphStore](../generated/torch_geometric.data.GraphStore.html#torch_geometric.data.GraphStore)) data object.
- **num_neighbors** (*List**[*[int](https://docs.python.org/3/library/functions.html#int)*] or **Dict**[**Tuple**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*]**, **List**[*[int](https://docs.python.org/3/library/functions.html#int)*]**]*) – The number of neighbors to sample for each node in each iteration.
If an entry is set to `-1`, all neighbors will be included.
In heterogeneous graphs, may also take in a dictionary denoting
the amount of neighbors to sample for each individual edge type.
- **master_addr** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – RPC address for distributed loader communication,
*i.e.* the IP address of the master node.
- **master_port** (*Union**[*[int](https://docs.python.org/3/library/functions.html#int)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) – Open port for RPC communication with
the master node.
- **current_ctx** (DistContext) – Distributed context information of the
current process.
- **concurrency** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – RPC concurrency used for defining the
maximum size of the asynchronous processing queue.
(default: `1`)


All other arguments follow the interface of
[torch_geometric.loader.NeighborLoader](loader.html#torch_geometric.loader.NeighborLoader).


***class *DistLinkNeighborLoader(*data: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[LocalFeatureStore, LocalGraphStore]*, *num_neighbors: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]]*, *master_addr: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *master_port: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [str](https://docs.python.org/3/library/stdtypes.html#str)]*, *current_ctx: DistContext*, *edge_label_index: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [None](https://docs.python.org/3/library/constants.html#None), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]] = None*, *edge_label: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *edge_label_time: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *dist_sampler: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[DistNeighborSampler] = None*, *replace: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *subgraph_type: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[SubgraphType, [str](https://docs.python.org/3/library/stdtypes.html#str)] = 'directional'*, *disjoint: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *temporal_strategy: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'uniform'*, *neg_sampling: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[NegativeSampling](sampler.html#torch_geometric.sampler.NegativeSampling)] = None*, *neg_sampling_ratio: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)]] = None*, *time_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *transform: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)] = None*, *concurrency: [int](https://docs.python.org/3/library/functions.html#int) = 1*, *num_rpc_threads: [int](https://docs.python.org/3/library/functions.html#int) = 16*, *filter_per_worker: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[bool](https://docs.python.org/3/library/functions.html#bool)] = False*, *async_sampling: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, *device: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[device](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.device)] = None*, ***kwargs*)[[source]](../_modules/torch_geometric/distributed/dist_link_neighbor_loader.html#DistLinkNeighborLoader)**
: A distributed loader that performs sampling from edges.


**Parameters:**
: - **data** ([tuple](https://docs.python.org/3/library/stdtypes.html#tuple)) – A ([FeatureStore](../generated/torch_geometric.data.FeatureStore.html#torch_geometric.data.FeatureStore),
[GraphStore](../generated/torch_geometric.data.GraphStore.html#torch_geometric.data.GraphStore)) data object.
- **num_neighbors** (*List**[*[int](https://docs.python.org/3/library/functions.html#int)*] or **Dict**[**Tuple**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*]**, **List**[*[int](https://docs.python.org/3/library/functions.html#int)*]**]*) – The number of neighbors to sample for each node in each iteration.
If an entry is set to `-1`, all neighbors will be included.
In heterogeneous graphs, may also take in a dictionary denoting
the amount of neighbors to sample for each individual edge type.
- **master_addr** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – RPC address for distributed loader communication,
*i.e.* the IP address of the master node.
- **master_port** (*Union**[*[int](https://docs.python.org/3/library/functions.html#int)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*]*) – Open port for RPC communication with
the master node.
- **current_ctx** (DistContext) – Distributed context information of the
current process.
- **concurrency** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – RPC concurrency used for defining the
maximum size of the asynchronous processing queue.
(default: `1`)


All other arguments follow the interface of
[torch_geometric.loader.LinkNeighborLoader](loader.html#torch_geometric.loader.LinkNeighborLoader).


