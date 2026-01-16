| DataLoader | A data loader which merges data objects from atorch_geometric.data.Datasetto a mini-batch. |
| --- | --- |
| NodeLoader | A data loader that performs mini-batch sampling from node information, using a genericBaseSamplerimplementation that defines asample_from_nodes()function and is supported on the provided inputdataobject. |
| LinkLoader | A data loader that performs mini-batch sampling from link information, using a genericBaseSamplerimplementation that defines asample_from_edges()function and is supported on the provided inputdataobject. |
| NeighborLoader | A data loader that performs neighbor sampling as introduced in the"Inductive Representation Learning on Large Graphs"paper. |
| LinkNeighborLoader | A link-based data loader derived as an extension of the node-basedtorch_geometric.loader.NeighborLoader. |
| HGTLoader | The Heterogeneous Graph Sampler from the"Heterogeneous Graph Transformer"paper. |
| ClusterData | Clusters/partitions a graph data object into multiple subgraphs, as motivated by the"Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks"paper. |
| ClusterLoader | The data loader scheme from the"Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks"paper which merges partitioned subgraphs and their between-cluster links from a large-scale graph data object to form a mini-batch. |
| GraphSAINTSampler | The GraphSAINT sampler base class from the"GraphSAINT: Graph Sampling Based Inductive Learning Method"paper. |
| GraphSAINTNodeSampler | The GraphSAINT node sampler class (seeGraphSAINTSampler). |
| GraphSAINTEdgeSampler | The GraphSAINT edge sampler class (seeGraphSAINTSampler). |
| GraphSAINTRandomWalkSampler | The GraphSAINT random walk sampler class (seeGraphSAINTSampler). |
| ShaDowKHopSampler | The ShaDow$k$-hop sampler from the"Decoupling the Depth and Scope of Graph Neural Networks"paper. |
| RandomNodeLoader | A data loader that randomly samples nodes within a graph and returns their induced subgraph. |
| ZipLoader | A loader that returns a tuple of data objects by sampling from multipleNodeLoaderorLinkLoaderinstances. |
| DataListLoader | A data loader which batches data objects from atorch_geometric.data.datasetto aPythonlist. |
| DenseDataLoader | A data loader which batches data objects from atorch_geometric.data.datasetto atorch_geometric.data.Batchobject by stacking all attributes in a new dimension. |
| TemporalDataLoader | A data loader which merges successive events of atorch_geometric.data.TemporalDatato a mini-batch. |
| NeighborSampler | The neighbor sampler from the"Inductive Representation Learning on Large Graphs"paper, which allows for mini-batch training of GNNs on large-scale graphs where full-batch training is not feasible. |
| ImbalancedSampler | A weighted random sampler that randomly samples elements according to class distribution. |
| DynamicBatchSampler | Dynamically adds samples to a mini-batch up to a maximum size (either based on number of nodes or number of edges). |
| PrefetchLoader | A GPU prefetcher class for asynchronously transferring data of atorch.utils.data.DataLoaderfrom host memory to device memory. |
| CachedLoader | A loader to cache mini-batch outputs, e.g., obtained duringNeighborLoaderiterations. |
| AffinityMixin | A context manager to enable CPU affinity for data loader workers (only used when running on CPU devices). |


***class *DataLoader(*dataset: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Dataset](../generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset), [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[BaseData], DatasetAdapter]*, *batch_size: [int](https://docs.python.org/3/library/functions.html#int) = 1*, *shuffle: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *follow_batch: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*, *exclude_keys: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*, ***kwargs*)[[source]](../_modules/torch_geometric/loader/dataloader.html#DataLoader)**
: A data loader which merges data objects from a
[torch_geometric.data.Dataset](../generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset) to a mini-batch.
Data objects can be either of type [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) or
[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData).


**Parameters:**
: - **dataset** ([Dataset](../generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset)) – The dataset from which to load the data.
- **batch_size** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – How many samples per batch to load.
(default: `1`)
- **shuffle** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), the data will be
reshuffled at every epoch. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **follow_batch** (*List**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]**, **optional*) – Creates assignment batch
vectors for each key in the list. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **exclude_keys** (*List**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]**, **optional*) – Will exclude each key in the
list. (default: [None](https://docs.python.org/3/library/constants.html#None))
- ****kwargs** (*optional*) – Additional arguments of
[torch.utils.data.DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader).


***class *NodeLoader(*data: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[FeatureStore](../generated/torch_geometric.data.FeatureStore.html#torch_geometric.data.FeatureStore), [GraphStore](../generated/torch_geometric.data.GraphStore.html#torch_geometric.data.GraphStore)]]*, *node_sampler: [BaseSampler](sampler.html#torch_geometric.sampler.BaseSampler)*, *input_nodes: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [None](https://docs.python.org/3/library/constants.html#None), [str](https://docs.python.org/3/library/stdtypes.html#str), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]] = None*, *input_time: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *transform: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)] = None*, *transform_sampler_output: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)] = None*, *filter_per_worker: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[bool](https://docs.python.org/3/library/functions.html#bool)] = None*, *custom_cls: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)] = None*, *input_id: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, ***kwargs*)[[source]](../_modules/torch_geometric/loader/node_loader.html#NodeLoader)**
: A data loader that performs mini-batch sampling from node information,
using a generic [BaseSampler](sampler.html#torch_geometric.sampler.BaseSampler)
implementation that defines a
[sample_from_nodes()](sampler.html#torch_geometric.sampler.BaseSampler.sample_from_nodes) function and
is supported on the provided input `data` object.


**Parameters:**
: - **data** (*Any*) – A [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data),
[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData), or
([FeatureStore](../generated/torch_geometric.data.FeatureStore.html#torch_geometric.data.FeatureStore),
[GraphStore](../generated/torch_geometric.data.GraphStore.html#torch_geometric.data.GraphStore)) data object.
- **node_sampler** ([torch_geometric.sampler.BaseSampler](sampler.html#torch_geometric.sampler.BaseSampler)) – The sampler
implementation to be used with this loader.
Needs to implement
[sample_from_nodes()](sampler.html#torch_geometric.sampler.BaseSampler.sample_from_nodes).
The sampler implementation must be compatible with the input
`data` object.
- **input_nodes** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)* or *[str](https://docs.python.org/3/library/stdtypes.html#str)* or **Tuple**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]*) – The
indices of seed nodes to start sampling from.
Needs to be either given as a `torch.LongTensor` or
`torch.BoolTensor`.
If set to [None](https://docs.python.org/3/library/constants.html#None), all nodes will be considered.
In heterogeneous graphs, needs to be passed as a tuple that holds
the node type and node indices. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **input_time** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – Optional values to override the
timestamp for the input nodes given in `input_nodes`. If not
set, will use the timestamps in `time_attr` as default (if
present). The `time_attr` needs to be set for this to work.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **transform** (*callable**, **optional*) – A function/transform that takes in
a sampled mini-batch and returns a transformed version.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **transform_sampler_output** (*callable**, **optional*) – A function/transform
that takes in a [torch_geometric.sampler.SamplerOutput](sampler.html#torch_geometric.sampler.SamplerOutput) and
returns a transformed version. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **filter_per_worker** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will filter
the returned data in each worker’s subprocess.
If set to [False](https://docs.python.org/3/library/constants.html#False), will filter the returned data in the main
process.
If set to [None](https://docs.python.org/3/library/constants.html#None), will automatically infer the decision based
on whether data partially lives on the GPU
(`filter_per_worker=True`) or entirely on the CPU
(`filter_per_worker=False`).
There exists different trade-offs for setting this option.
Specifically, setting this option to [True](https://docs.python.org/3/library/constants.html#True) for in-memory
datasets will move all features to shared memory, which may result
in too many open file handles. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **custom_cls** ([HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)*, **optional*) – A custom
[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData) class to return for
mini-batches in case of remote backends. (default: [None](https://docs.python.org/3/library/constants.html#None))
- ****kwargs** (*optional*) – Additional arguments of
[torch.utils.data.DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader), such as `batch_size`,
`shuffle`, `drop_last` or `num_workers`.


**collate_fn(*index: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)[[source]](../_modules/torch_geometric/loader/node_loader.html#NodeLoader.collate_fn)**
: Samples a subgraph from a batch of input nodes.


**Return type:**
: [Any](https://docs.python.org/3/library/typing.html#typing.Any)


**filter_fn(*out: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[SamplerOutput](sampler.html#torch_geometric.sampler.SamplerOutput), [HeteroSamplerOutput](sampler.html#torch_geometric.sampler.HeteroSamplerOutput)]*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)][[source]](../_modules/torch_geometric/loader/node_loader.html#NodeLoader.filter_fn)**
: Joins the sampled nodes with their corresponding features,
returning the resulting [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) or
[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData) object to be used downstream.


**Return type:**
: `Union`[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)]


***class *LinkLoader(*data: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[FeatureStore](../generated/torch_geometric.data.FeatureStore.html#torch_geometric.data.FeatureStore), [GraphStore](../generated/torch_geometric.data.GraphStore.html#torch_geometric.data.GraphStore)]]*, *link_sampler: [BaseSampler](sampler.html#torch_geometric.sampler.BaseSampler)*, *edge_label_index: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [None](https://docs.python.org/3/library/constants.html#None), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]] = None*, *edge_label: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *edge_label_time: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *neg_sampling: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[NegativeSampling](sampler.html#torch_geometric.sampler.NegativeSampling)] = None*, *neg_sampling_ratio: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)]] = None*, *transform: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)] = None*, *transform_sampler_output: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)] = None*, *filter_per_worker: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[bool](https://docs.python.org/3/library/functions.html#bool)] = None*, *custom_cls: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)] = None*, *input_id: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, ***kwargs*)[[source]](../_modules/torch_geometric/loader/link_loader.html#LinkLoader)**
: A data loader that performs mini-batch sampling from link information,
using a generic [BaseSampler](sampler.html#torch_geometric.sampler.BaseSampler)
implementation that defines a
[sample_from_edges()](sampler.html#torch_geometric.sampler.BaseSampler.sample_from_edges) function and
is supported on the provided input `data` object.


> **Note:** Negative sampling is currently implemented in an approximate
way, *i.e.* negative edges may contain false negatives.


**Parameters:**
: - **data** (*Any*) – A [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data),
[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData), or
([FeatureStore](../generated/torch_geometric.data.FeatureStore.html#torch_geometric.data.FeatureStore),
[GraphStore](../generated/torch_geometric.data.GraphStore.html#torch_geometric.data.GraphStore)) data object.
- **link_sampler** ([torch_geometric.sampler.BaseSampler](sampler.html#torch_geometric.sampler.BaseSampler)) – The sampler
implementation to be used with this loader.
Needs to implement
[sample_from_edges()](sampler.html#torch_geometric.sampler.BaseSampler.sample_from_edges).
The sampler implementation must be compatible with the input
`data` object.
- **edge_label_index** (*Tensor** or **EdgeType** or **Tuple**[**EdgeType**, **Tensor**]*) – The edge indices, holding source and destination nodes to start
sampling from.
If set to [None](https://docs.python.org/3/library/constants.html#None), all edges will be considered.
In heterogeneous graphs, needs to be passed as a tuple that holds
the edge type and corresponding edge indices.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **edge_label** (*Tensor**, **optional*) – The labels of edge indices from which to
start sampling from. Must be the same length as
the `edge_label_index`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **edge_label_time** (*Tensor**, **optional*) – The timestamps of edge indices from
which to start sampling from. Must be the same length as
`edge_label_index`. If set, temporal sampling will be
used such that neighbors are guaranteed to fulfill temporal
constraints, *i.e.*, neighbors have an earlier timestamp than
the output edge. The `time_attr` needs to be set for this
to work. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **neg_sampling** ([NegativeSampling](sampler.html#torch_geometric.sampler.NegativeSampling)*, **optional*) – The negative sampling
configuration.
For negative sampling mode `"binary"`, samples can be accessed
via the attributes `edge_label_index` and `edge_label` in
the respective edge type of the returned mini-batch.
In case `edge_label` does not exist, it will be automatically
created and represents a binary classification task (`0` =
negative edge, `1` = positive edge).
In case `edge_label` does exist, it has to be a categorical
label from `0` to `num_classes - 1`.
After negative sampling, label `0` represents negative edges,
and labels `1` to `num_classes` represent the labels of
positive edges.
Note that returned labels are of type `torch.float` for binary
classification (to facilitate the ease-of-use of
`F.binary_cross_entropy()`) and of type
`torch.long` for multi-class classification (to facilitate the
ease-of-use of `F.cross_entropy()`).
For negative sampling mode `"triplet"`, samples can be
accessed via the attributes `src_index`, `dst_pos_index`
and `dst_neg_index` in the respective node types of the
returned mini-batch.
`edge_label` needs to be [None](https://docs.python.org/3/library/constants.html#None) for `"triplet"`
negative sampling mode.
If set to [None](https://docs.python.org/3/library/constants.html#None), no negative sampling strategy is applied.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **neg_sampling_ratio** ([int](https://docs.python.org/3/library/functions.html#int)* or *[float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – The ratio of sampled
negative edges to the number of positive edges.
Deprecated in favor of the `neg_sampling` argument.
(default: [None](https://docs.python.org/3/library/constants.html#None)).
- **transform** (*callable**, **optional*) – A function/transform that takes in
a sampled mini-batch and returns a transformed version.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **transform_sampler_output** (*callable**, **optional*) – A function/transform
that takes in a [torch_geometric.sampler.SamplerOutput](sampler.html#torch_geometric.sampler.SamplerOutput) and
returns a transformed version. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **filter_per_worker** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will filter
the returned data in each worker’s subprocess.
If set to [False](https://docs.python.org/3/library/constants.html#False), will filter the returned data in the main
process.
If set to [None](https://docs.python.org/3/library/constants.html#None), will automatically infer the decision based
on whether data partially lives on the GPU
(`filter_per_worker=True`) or entirely on the CPU
(`filter_per_worker=False`).
There exists different trade-offs for setting this option.
Specifically, setting this option to [True](https://docs.python.org/3/library/constants.html#True) for in-memory
datasets will move all features to shared memory, which may result
in too many open file handles. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **custom_cls** ([HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)*, **optional*) – A custom
[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData) class to return for
mini-batches in case of remote backends. (default: [None](https://docs.python.org/3/library/constants.html#None))
- ****kwargs** (*optional*) – Additional arguments of
[torch.utils.data.DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader), such as `batch_size`,
`shuffle`, `drop_last` or `num_workers`.


**collate_fn(*index: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)[[source]](../_modules/torch_geometric/loader/link_loader.html#LinkLoader.collate_fn)**
: Samples a subgraph from a batch of input edges.


**Return type:**
: [Any](https://docs.python.org/3/library/typing.html#typing.Any)


**filter_fn(*out: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[SamplerOutput](sampler.html#torch_geometric.sampler.SamplerOutput), [HeteroSamplerOutput](sampler.html#torch_geometric.sampler.HeteroSamplerOutput)]*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)][[source]](../_modules/torch_geometric/loader/link_loader.html#LinkLoader.filter_fn)**
: Joins the sampled nodes with their corresponding features,
returning the resulting [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) or
[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData) object to be used downstream.


**Return type:**
: `Union`[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)]


***class *NeighborLoader(*data: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[FeatureStore](../generated/torch_geometric.data.FeatureStore.html#torch_geometric.data.FeatureStore), [GraphStore](../generated/torch_geometric.data.GraphStore.html#torch_geometric.data.GraphStore)]]*, *num_neighbors: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]]*, *input_nodes: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [None](https://docs.python.org/3/library/constants.html#None), [str](https://docs.python.org/3/library/stdtypes.html#str), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]] = None*, *input_time: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *replace: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *subgraph_type: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[SubgraphType, [str](https://docs.python.org/3/library/stdtypes.html#str)] = 'directional'*, *disjoint: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *temporal_strategy: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'uniform'*, *time_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *weight_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *transform: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)] = None*, *transform_sampler_output: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)] = None*, *is_sorted: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *filter_per_worker: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[bool](https://docs.python.org/3/library/functions.html#bool)] = None*, *neighbor_sampler: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[NeighborSampler](sampler.html#torch_geometric.sampler.NeighborSampler)] = None*, *directed: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, ***kwargs*)[[source]](../_modules/torch_geometric/loader/neighbor_loader.html#NeighborLoader)**
: A data loader that performs neighbor sampling as introduced in the
[“Inductive Representation Learning on Large Graphs”](https://arxiv.org/abs/1706.02216) paper.
This loader allows for mini-batch training of GNNs on large-scale graphs
where full-batch training is not feasible.


More specifically, `num_neighbors` denotes how many neighbors are
sampled for each node in each iteration.
NeighborLoader takes in this list of
`num_neighbors` and iteratively samples `num_neighbors[i]` for
each node involved in iteration `i - 1`.


Sampled nodes are sorted based on the order in which they were sampled.
In particular, the first `batch_size` nodes represent the set of
original mini-batch nodes.


```
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader

data = Planetoid(path, name='Cora')[0]

loader = NeighborLoader(
    data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=128,
    input_nodes=data.train_mask,
)

sampled_data = next(iter(loader))
print(sampled_data.batch_size)
>>> 128
```


By default, the data loader will only include the edges that were
originally sampled (`directed = True`).
This option should only be used in case the number of hops is equivalent to
the number of GNN layers.
In case the number of GNN layers is greater than the number of hops,
consider setting `directed = False`, which will include all edges
between all sampled nodes (but is slightly slower as a result).


Furthermore, NeighborLoader works for both
**homogeneous** graphs stored via [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) as
well as **heterogeneous** graphs stored via
[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData).
When operating in heterogeneous graphs, up to `num_neighbors`
neighbors will be sampled for each `edge_type`.
However, more fine-grained control over
the amount of sampled neighbors of individual edge types is possible:


```
from torch_geometric.datasets import OGB_MAG
from torch_geometric.loader import NeighborLoader

hetero_data = OGB_MAG(path)[0]

loader = NeighborLoader(
    hetero_data,
    # Sample 30 neighbors for each node and edge type for 2 iterations
    num_neighbors={key: [30] * 2 for key in hetero_data.edge_types},
    # Use a batch size of 128 for sampling training nodes of type paper
    batch_size=128,
    input_nodes=('paper', hetero_data['paper'].train_mask),
)

sampled_hetero_data = next(iter(loader))
print(sampled_hetero_data['paper'].batch_size)
>>> 128
```


> **Note:** For an example of using
NeighborLoader, see
[examples/hetero/to_hetero_mag.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py).


The NeighborLoader will return subgraphs
where global node indices are mapped to local indices corresponding to this
specific subgraph. However, often times it is desired to map the nodes of
the current subgraph back to the global node indices. The
NeighborLoader will include this mapping
as part of the `data` object:


```
loader = NeighborLoader(data, ...)
sampled_data = next(iter(loader))
print(sampled_data.n_id)  # Global node index of each node in batch.
```


In particular, the data loader will add the following attributes to the
returned mini-batch:


- `batch_size` The number of seed nodes (first nodes in the batch)
- `n_id` The global node index for every sampled node
- `e_id` The global edge index for every sampled edge
- `input_id`: The global index of the `input_nodes`
- `num_sampled_nodes`: The number of sampled nodes in each hop
- `num_sampled_edges`: The number of sampled edges in each hop


**Parameters:**
: - **data** (*Any*) – A [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data),
[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData), or
([FeatureStore](../generated/torch_geometric.data.FeatureStore.html#torch_geometric.data.FeatureStore),
[GraphStore](../generated/torch_geometric.data.GraphStore.html#torch_geometric.data.GraphStore)) data object.
- **num_neighbors** (*List**[*[int](https://docs.python.org/3/library/functions.html#int)*] or **Dict**[**Tuple**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*]**, **List**[*[int](https://docs.python.org/3/library/functions.html#int)*]**]*) – The
number of neighbors to sample for each node in each iteration.
If an entry is set to `-1`, all neighbors will be included.
In heterogeneous graphs, may also take in a dictionary denoting
the amount of neighbors to sample for each individual edge type.
- **input_nodes** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)* or *[str](https://docs.python.org/3/library/stdtypes.html#str)* or **Tuple**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]*) – The
indices of nodes for which neighbors are sampled to create
mini-batches.
Needs to be either given as a `torch.LongTensor` or
`torch.BoolTensor`.
If set to [None](https://docs.python.org/3/library/constants.html#None), all nodes will be considered.
In heterogeneous graphs, needs to be passed as a tuple that holds
the node type and node indices. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **input_time** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – Optional values to override the
timestamp for the input nodes given in `input_nodes`. If not
set, will use the timestamps in `time_attr` as default (if
present). The `time_attr` needs to be set for this to work.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **replace** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will sample with
replacement. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **subgraph_type** (*SubgraphType** or *[str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The type of the returned
subgraph.
If set to `"directional"`, the returned subgraph only holds
the sampled (directed) edges which are necessary to compute
representations for the sampled seed nodes.
If set to `"bidirectional"`, sampled edges are converted to
bidirectional edges.
If set to `"induced"`, the returned subgraph contains the
induced subgraph of all sampled nodes.
(default: `"directional"`)
- **disjoint** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to :obj: True, each seed node will
create its own disjoint subgraph.
If set to [True](https://docs.python.org/3/library/constants.html#True), mini-batch outputs will have a `batch`
vector holding the mapping of nodes to their respective subgraph.
Will get automatically set to [True](https://docs.python.org/3/library/constants.html#True) in case of temporal
sampling. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **temporal_strategy** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The sampling strategy when using
temporal sampling (`"uniform"`, `"last"`).
If set to `"uniform"`, will sample uniformly across neighbors
that fulfill temporal constraints.
If set to `"last"`, will sample the last num_neighbors that
fulfill temporal constraints.
(default: `"uniform"`)
- **time_attr** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The name of the attribute that denotes
timestamps for either the nodes or edges in the graph.
If set, temporal sampling will be used such that neighbors are
guaranteed to fulfill temporal constraints, *i.e.* neighbors have
an earlier or equal timestamp than the center node.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **weight_attr** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The name of the attribute that denotes
edge weights in the graph.
If set, weighted/biased sampling will be used such that neighbors
are more likely to get sampled the higher their edge weights are.
Edge weights do not need to sum to one, but must be non-negative,
finite and have a non-zero sum within local neighborhoods.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **transform** (*callable**, **optional*) – A function/transform that takes in
a sampled mini-batch and returns a transformed version.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **transform_sampler_output** (*callable**, **optional*) – A function/transform
that takes in a [torch_geometric.sampler.SamplerOutput](sampler.html#torch_geometric.sampler.SamplerOutput) and
returns a transformed version. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **is_sorted** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), assumes that
`edge_index` is sorted by column.
If `time_attr` is set, additionally requires that rows are
sorted according to time within individual neighborhoods.
This avoids internal re-sorting of the data and can improve
runtime and memory efficiency. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **filter_per_worker** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will filter
the returned data in each worker’s subprocess.
If set to [False](https://docs.python.org/3/library/constants.html#False), will filter the returned data in the main
process.
If set to [None](https://docs.python.org/3/library/constants.html#None), will automatically infer the decision based
on whether data partially lives on the GPU
(`filter_per_worker=True`) or entirely on the CPU
(`filter_per_worker=False`).
There exists different trade-offs for setting this option.
Specifically, setting this option to [True](https://docs.python.org/3/library/constants.html#True) for in-memory
datasets will move all features to shared memory, which may result
in too many open file handles. (default: [None](https://docs.python.org/3/library/constants.html#None))
- ****kwargs** (*optional*) – Additional arguments of
[torch.utils.data.DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader), such as `batch_size`,
`shuffle`, `drop_last` or `num_workers`.


***class *LinkNeighborLoader(*data: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[FeatureStore](../generated/torch_geometric.data.FeatureStore.html#torch_geometric.data.FeatureStore), [GraphStore](../generated/torch_geometric.data.GraphStore.html#torch_geometric.data.GraphStore)]]*, *num_neighbors: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]]*, *edge_label_index: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [None](https://docs.python.org/3/library/constants.html#None), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]] = None*, *edge_label: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *edge_label_time: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *replace: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *subgraph_type: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[SubgraphType, [str](https://docs.python.org/3/library/stdtypes.html#str)] = 'directional'*, *disjoint: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *temporal_strategy: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'uniform'*, *neg_sampling: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[NegativeSampling](sampler.html#torch_geometric.sampler.NegativeSampling)] = None*, *neg_sampling_ratio: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)]] = None*, *time_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *weight_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *transform: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)] = None*, *transform_sampler_output: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)] = None*, *is_sorted: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *filter_per_worker: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[bool](https://docs.python.org/3/library/functions.html#bool)] = None*, *neighbor_sampler: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[NeighborSampler](sampler.html#torch_geometric.sampler.NeighborSampler)] = None*, *directed: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, ***kwargs*)[[source]](../_modules/torch_geometric/loader/link_neighbor_loader.html#LinkNeighborLoader)**
: A link-based data loader derived as an extension of the node-based
torch_geometric.loader.NeighborLoader.
This loader allows for mini-batch training of GNNs on large-scale graphs
where full-batch training is not feasible.


More specifically, this loader first selects a sample of edges from the
set of input edges `edge_label_index` (which may or not be edges in
the original graph) and then constructs a subgraph from all the nodes
present in this list by sampling `num_neighbors` neighbors in each
iteration.


```
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import LinkNeighborLoader

data = Planetoid(path, name='Cora')[0]

loader = LinkNeighborLoader(
    data,
    # Sample 30 neighbors for each node for 2 iterations
    num_neighbors=[30] * 2,
    # Use a batch size of 128 for sampling training nodes
    batch_size=128,
    edge_label_index=data.edge_index,
)

sampled_data = next(iter(loader))
print(sampled_data)
>>> Data(x=[1368, 1433], edge_index=[2, 3103], y=[1368],
         train_mask=[1368], val_mask=[1368], test_mask=[1368],
         edge_label_index=[2, 128])
```


It is additionally possible to provide edge labels for sampled edges, which
are then added to the batch:


```
loader = LinkNeighborLoader(
    data,
    num_neighbors=[30] * 2,
    batch_size=128,
    edge_label_index=data.edge_index,
    edge_label=torch.ones(data.edge_index.size(1))
)

sampled_data = next(iter(loader))
print(sampled_data)
>>> Data(x=[1368, 1433], edge_index=[2, 3103], y=[1368],
         train_mask=[1368], val_mask=[1368], test_mask=[1368],
         edge_label_index=[2, 128], edge_label=[128])
```


The rest of the functionality mirrors that of
NeighborLoader, including support for
heterogeneous graphs.
In particular, the data loader will add the following attributes to the
returned mini-batch:


- `n_id` The global node index for every sampled node
- `e_id` The global edge index for every sampled edge
- `input_id`: The global index of the `edge_label_index`
- `num_sampled_nodes`: The number of sampled nodes in each hop
- `num_sampled_edges`: The number of sampled edges in each hop


> **Note:** Negative sampling is currently implemented in an approximate
way, *i.e.* negative edges may contain false negatives.


> **Warning:** Note that the sampling scheme is independent from the edge we are
making a prediction for.
That is, by default supervision edges in `edge_label_index`
**will not** get masked out during sampling.
In case there exists an overlap between message passing edges in
`data.edge_index` and supervision edges in
`edge_label_index`, you might end up sampling an edge you are
making a prediction for.
You can generally avoid this behavior (if desired) by making
`data.edge_index` and `edge_label_index` two disjoint sets of
edges, *e.g.*, via the
[RandomLinkSplit](../generated/torch_geometric.transforms.RandomLinkSplit.html#torch_geometric.transforms.RandomLinkSplit) transformation and
its `disjoint_train_ratio` argument.


**Parameters:**
: - **data** (*Any*) – A [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data),
[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData), or
([FeatureStore](../generated/torch_geometric.data.FeatureStore.html#torch_geometric.data.FeatureStore),
[GraphStore](../generated/torch_geometric.data.GraphStore.html#torch_geometric.data.GraphStore)) data object.
- **num_neighbors** (*List**[*[int](https://docs.python.org/3/library/functions.html#int)*] or **Dict**[**Tuple**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*]**, **List**[*[int](https://docs.python.org/3/library/functions.html#int)*]**]*) – The
number of neighbors to sample for each node in each iteration.
If an entry is set to `-1`, all neighbors will be included.
In heterogeneous graphs, may also take in a dictionary denoting
the amount of neighbors to sample for each individual edge type.
- **edge_label_index** (*Tensor** or **EdgeType** or **Tuple**[**EdgeType**, **Tensor**]*) – The edge indices for which neighbors are sampled to create
mini-batches.
If set to [None](https://docs.python.org/3/library/constants.html#None), all edges will be considered.
In heterogeneous graphs, needs to be passed as a tuple that holds
the edge type and corresponding edge indices.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **edge_label** (*Tensor**, **optional*) – The labels of edge indices for
which neighbors are sampled. Must be the same length as
the `edge_label_index`. If set to [None](https://docs.python.org/3/library/constants.html#None) its set to
torch.zeros(…) internally. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **edge_label_time** (*Tensor**, **optional*) – The timestamps for edge indices
for which neighbors are sampled. Must be the same length as
`edge_label_index`. If set, temporal sampling will be
used such that neighbors are guaranteed to fulfill temporal
constraints, *i.e.*, neighbors have an earlier timestamp than
the output edge. The `time_attr` needs to be set for this
to work. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **replace** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will sample with
replacement. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **subgraph_type** (*SubgraphType** or *[str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The type of the returned
subgraph.
If set to `"directional"`, the returned subgraph only holds
the sampled (directed) edges which are necessary to compute
representations for the sampled seed nodes.
If set to `"bidirectional"`, sampled edges are converted to
bidirectional edges.
If set to `"induced"`, the returned subgraph contains the
induced subgraph of all sampled nodes.
(default: `"directional"`)
- **disjoint** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to :obj: True, each seed node will
create its own disjoint subgraph.
If set to [True](https://docs.python.org/3/library/constants.html#True), mini-batch outputs will have a `batch`
vector holding the mapping of nodes to their respective subgraph.
Will get automatically set to [True](https://docs.python.org/3/library/constants.html#True) in case of temporal
sampling. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **temporal_strategy** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The sampling strategy when using
temporal sampling (`"uniform"`, `"last"`).
If set to `"uniform"`, will sample uniformly across neighbors
that fulfill temporal constraints.
If set to `"last"`, will sample the last num_neighbors that
fulfill temporal constraints.
(default: `"uniform"`)
- **neg_sampling** ([NegativeSampling](sampler.html#torch_geometric.sampler.NegativeSampling)*, **optional*) – The negative sampling
configuration.
For negative sampling mode `"binary"`, samples can be accessed
via the attributes `edge_label_index` and `edge_label` in
the respective edge type of the returned mini-batch.
In case `edge_label` does not exist, it will be automatically
created and represents a binary classification task (`0` =
negative edge, `1` = positive edge).
In case `edge_label` does exist, it has to be a categorical
label from `0` to `num_classes - 1`.
After negative sampling, label `0` represents negative edges,
and labels `1` to `num_classes` represent the labels of
positive edges.
Note that returned labels are of type `torch.float` for binary
classification (to facilitate the ease-of-use of
`F.binary_cross_entropy()`) and of type
`torch.long` for multi-class classification (to facilitate the
ease-of-use of `F.cross_entropy()`).
For negative sampling mode `"triplet"`, samples can be
accessed via the attributes `src_index`, `dst_pos_index`
and `dst_neg_index` in the respective node types of the
returned mini-batch.
`edge_label` needs to be [None](https://docs.python.org/3/library/constants.html#None) for `"triplet"`
negative sampling mode.
If set to [None](https://docs.python.org/3/library/constants.html#None), no negative sampling strategy is applied.
(default: [None](https://docs.python.org/3/library/constants.html#None))
For example use obj:neg_sampling=dict(mode= ‘binary’, amount=0.5)
- **neg_sampling_ratio** ([int](https://docs.python.org/3/library/functions.html#int)* or *[float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – The ratio of sampled
negative edges to the number of positive edges.
Deprecated in favor of the `neg_sampling` argument.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **time_attr** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The name of the attribute that denotes
timestamps for either the nodes or edges in the graph.
If set, temporal sampling will be used such that neighbors are
guaranteed to fulfill temporal constraints, *i.e.* neighbors have
an earlier or equal timestamp than the center node.
Only used if `edge_label_time` is set. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **weight_attr** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The name of the attribute that denotes
edge weights in the graph.
If set, weighted/biased sampling will be used such that neighbors
are more likely to get sampled the higher their edge weights are.
Edge weights do not need to sum to one, but must be non-negative,
finite and have a non-zero sum within local neighborhoods.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **transform** (*callable**, **optional*) – A function/transform that takes in
a sampled mini-batch and returns a transformed version.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **transform_sampler_output** (*callable**, **optional*) – A function/transform
that takes in a [torch_geometric.sampler.SamplerOutput](sampler.html#torch_geometric.sampler.SamplerOutput) and
returns a transformed version. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **is_sorted** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), assumes that
`edge_index` is sorted by column.
If `time_attr` is set, additionally requires that rows are
sorted according to time within individual neighborhoods.
This avoids internal re-sorting of the data and can improve
runtime and memory efficiency. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **filter_per_worker** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will filter
the returned data in each worker’s subprocess.
If set to [False](https://docs.python.org/3/library/constants.html#False), will filter the returned data in the main
process.
If set to [None](https://docs.python.org/3/library/constants.html#None), will automatically infer the decision based
on whether data partially lives on the GPU
(`filter_per_worker=True`) or entirely on the CPU
(`filter_per_worker=False`).
There exists different trade-offs for setting this option.
Specifically, setting this option to [True](https://docs.python.org/3/library/constants.html#True) for in-memory
datasets will move all features to shared memory, which may result
in too many open file handles. (default: [None](https://docs.python.org/3/library/constants.html#None))
- ****kwargs** (*optional*) – Additional arguments of
[torch.utils.data.DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader), such as `batch_size`,
`shuffle`, `drop_last` or `num_workers`.


***class *HGTLoader(*data: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[FeatureStore](../generated/torch_geometric.data.FeatureStore.html#torch_geometric.data.FeatureStore), [GraphStore](../generated/torch_geometric.data.GraphStore.html#torch_geometric.data.GraphStore)]]*, *num_samples: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]]*, *input_nodes: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]]*, *is_sorted: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *transform: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)] = None*, *transform_sampler_output: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)] = None*, *filter_per_worker: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[bool](https://docs.python.org/3/library/functions.html#bool)] = None*, ***kwargs*)[[source]](../_modules/torch_geometric/loader/hgt_loader.html#HGTLoader)**
: The Heterogeneous Graph Sampler from the [“Heterogeneous Graph
Transformer”](https://arxiv.org/abs/2003.01332) paper.
This loader allows for mini-batch training of GNNs on large-scale graphs
where full-batch training is not feasible.


`HGTLoader` tries to (1) keep a similar
number of nodes and edges for each type and (2) keep the sampled sub-graph
dense to minimize the information loss and reduce the sample variance.


Methodically, `HGTLoader` keeps track of a
node budget for each node type, which is then used to determine the
sampling probability of a node.
In particular, the probability of sampling a node is determined by the
number of connections to already sampled nodes and their node degrees.
With this, `HGTLoader` will sample a fixed
amount of neighbors for each node type in each iteration, as given by the
`num_samples` argument.


Sampled nodes are sorted based on the order in which they were sampled.
In particular, the first `batch_size` nodes represent the set of
original mini-batch nodes.


> **Note:** For an example of using `HGTLoader`, see
[examples/hetero/to_hetero_mag.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py).


```
from torch_geometric.loader import HGTLoader
from torch_geometric.datasets import OGB_MAG

hetero_data = OGB_MAG(path)[0]

loader = HGTLoader(
    hetero_data,
    # Sample 512 nodes per type and per iteration for 4 iterations
    num_samples={key: [512] * 4 for key in hetero_data.node_types},
    # Use a batch size of 128 for sampling training nodes of type paper
    batch_size=128,
    input_nodes=('paper', hetero_data['paper'].train_mask),
)

sampled_hetero_data = next(iter(loader))
print(sampled_data.batch_size)
>>> 128
```


**Parameters:**
: - **data** (*Any*) – A [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data),
[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData), or
([FeatureStore](../generated/torch_geometric.data.FeatureStore.html#torch_geometric.data.FeatureStore),
[GraphStore](../generated/torch_geometric.data.GraphStore.html#torch_geometric.data.GraphStore)) data object.
- **num_samples** (*List**[*[int](https://docs.python.org/3/library/functions.html#int)*] or **Dict**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, **List**[*[int](https://docs.python.org/3/library/functions.html#int)*]**]*) – The number of nodes to
sample in each iteration and for each node type.
If given as a list, will sample the same amount of nodes for each
node type.
- **input_nodes** ([str](https://docs.python.org/3/library/stdtypes.html#str)* or **Tuple**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]*) – The indices of nodes for
which neighbors are sampled to create mini-batches.
Needs to be passed as a tuple that holds the node type and
corresponding node indices.
Node indices need to be either given as a `torch.LongTensor`
or `torch.BoolTensor`.
If node indices are set to [None](https://docs.python.org/3/library/constants.html#None), all nodes of this specific
type will be considered.
- **transform** (*callable**, **optional*) – A function/transform that takes in
an a sampled mini-batch and returns a transformed version.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **transform_sampler_output** (*callable**, **optional*) – A function/transform
that takes in a [torch_geometric.sampler.SamplerOutput](sampler.html#torch_geometric.sampler.SamplerOutput) and
returns a transformed version. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **is_sorted** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), assumes that
`edge_index` is sorted by column. This avoids internal
re-sorting of the data and can improve runtime and memory
efficiency. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **filter_per_worker** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will filter
the returned data in each worker’s subprocess.
If set to [False](https://docs.python.org/3/library/constants.html#False), will filter the returned data in the main
process.
If set to [None](https://docs.python.org/3/library/constants.html#None), will automatically infer the decision based
on whether data partially lives on the GPU
(`filter_per_worker=True`) or entirely on the CPU
(`filter_per_worker=False`).
There exists different trade-offs for setting this option.
Specifically, setting this option to [True](https://docs.python.org/3/library/constants.html#True) for in-memory
datasets will move all features to shared memory, which may result
in too many open file handles. (default: [None](https://docs.python.org/3/library/constants.html#None))
- ****kwargs** (*optional*) – Additional arguments of
[torch.utils.data.DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader), such as `batch_size`,
`shuffle`, `drop_last` or `num_workers`.


***class *ClusterData(*data*, *num_parts: [int](https://docs.python.org/3/library/functions.html#int)*, *recursive: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *save_dir: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *filename: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *log: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, *keep_inter_cluster_edges: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *sparse_format: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['csr', 'csc'] = 'csr'*)[[source]](../_modules/torch_geometric/loader/cluster.html#ClusterData)**
: Clusters/partitions a graph data object into multiple subgraphs, as
motivated by the [“Cluster-GCN: An Efficient Algorithm for Training Deep
and Large Graph Convolutional Networks”](https://arxiv.org/abs/1905.07953) paper.


> **Note:** The underlying METIS algorithm requires undirected graphs as input.


**Parameters:**
: - **data** ([torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)) – The graph data object.
- **num_parts** ([int](https://docs.python.org/3/library/functions.html#int)) – The number of partitions.
- **recursive** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will use multilevel
recursive bisection instead of multilevel k-way partitioning.
(default: [False](https://docs.python.org/3/library/constants.html#False))
- **save_dir** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – If set, will save the partitioned data to the
`save_dir` directory for faster re-use. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **filename** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – Name of the stored partitioned file.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **log** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [False](https://docs.python.org/3/library/constants.html#False), will not log any
progress. (default: [True](https://docs.python.org/3/library/constants.html#True))
- **keep_inter_cluster_edges** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True),
will keep inter-cluster edge connections. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **sparse_format** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The sparse format to use for computing
partitions. (default: `"csr"`)


***class *ClusterLoader(*cluster_data*, ***kwargs*)[[source]](../_modules/torch_geometric/loader/cluster.html#ClusterLoader)**
: The data loader scheme from the [“Cluster-GCN: An Efficient Algorithm
for Training Deep and Large Graph Convolutional Networks”](https://arxiv.org/abs/1905.07953) paper which merges partitioned
subgraphs and their between-cluster links from a large-scale graph data
object to form a mini-batch.


> **Note:** Use ClusterData and
ClusterLoader in conjunction to
form mini-batches of clusters.
For an example of using Cluster-GCN, see
[examples/cluster_gcn_reddit.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/cluster_gcn_reddit.py) or
[examples/cluster_gcn_ppi.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/cluster_gcn_ppi.py).


**Parameters:**
: - **cluster_data** (torch_geometric.loader.ClusterData) – The already
partitioned data object.
- ****kwargs** (*optional*) – Additional arguments of
[torch.utils.data.DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader), such as `batch_size`,
`shuffle`, `drop_last` or `num_workers`.


***class *GraphSAINTSampler(*data*, *batch_size: [int](https://docs.python.org/3/library/functions.html#int)*, *num_steps: [int](https://docs.python.org/3/library/functions.html#int) = 1*, *sample_coverage: [int](https://docs.python.org/3/library/functions.html#int) = 0*, *save_dir: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *log: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, ***kwargs*)[[source]](../_modules/torch_geometric/loader/graph_saint.html#GraphSAINTSampler)**
: The GraphSAINT sampler base class from the [“GraphSAINT: Graph
Sampling Based Inductive Learning Method”](https://arxiv.org/abs/1907.04931) paper.
Given a graph in a `data` object, this class samples nodes and
constructs subgraphs that can be processed in a mini-batch fashion.
Normalization coefficients for each mini-batch are given via
`node_norm` and `edge_norm` data attributes.


> **Note:** See GraphSAINTNodeSampler,
GraphSAINTEdgeSampler and
GraphSAINTRandomWalkSampler for
currently supported samplers.
For an example of using GraphSAINT sampling, see
[examples/graph_saint.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/graph_saint.py).


**Parameters:**
: - **data** ([torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)) – The graph data object.
- **batch_size** ([int](https://docs.python.org/3/library/functions.html#int)) – The approximate number of samples per batch.
- **num_steps** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of iterations per epoch.
(default: `1`)
- **sample_coverage** ([int](https://docs.python.org/3/library/functions.html#int)) – How many samples per node should be used to
compute normalization statistics. (default: `0`)
- **save_dir** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – If set, will save normalization statistics to
the `save_dir` directory for faster re-use.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **log** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [False](https://docs.python.org/3/library/constants.html#False), will not log any
pre-processing progress. (default: [True](https://docs.python.org/3/library/constants.html#True))
- ****kwargs** (*optional*) – Additional arguments of
[torch.utils.data.DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader), such as `batch_size` or
`num_workers`.


***class *GraphSAINTNodeSampler(*data*, *batch_size: [int](https://docs.python.org/3/library/functions.html#int)*, *num_steps: [int](https://docs.python.org/3/library/functions.html#int) = 1*, *sample_coverage: [int](https://docs.python.org/3/library/functions.html#int) = 0*, *save_dir: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *log: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, ***kwargs*)[[source]](../_modules/torch_geometric/loader/graph_saint.html#GraphSAINTNodeSampler)**
: The GraphSAINT node sampler class (see
GraphSAINTSampler).


***class *GraphSAINTEdgeSampler(*data*, *batch_size: [int](https://docs.python.org/3/library/functions.html#int)*, *num_steps: [int](https://docs.python.org/3/library/functions.html#int) = 1*, *sample_coverage: [int](https://docs.python.org/3/library/functions.html#int) = 0*, *save_dir: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *log: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, ***kwargs*)[[source]](../_modules/torch_geometric/loader/graph_saint.html#GraphSAINTEdgeSampler)**
: The GraphSAINT edge sampler class (see
GraphSAINTSampler).


***class *GraphSAINTRandomWalkSampler(*data*, *batch_size: [int](https://docs.python.org/3/library/functions.html#int)*, *walk_length: [int](https://docs.python.org/3/library/functions.html#int)*, *num_steps: [int](https://docs.python.org/3/library/functions.html#int) = 1*, *sample_coverage: [int](https://docs.python.org/3/library/functions.html#int) = 0*, *save_dir: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *log: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, ***kwargs*)[[source]](../_modules/torch_geometric/loader/graph_saint.html#GraphSAINTRandomWalkSampler)**
: The GraphSAINT random walk sampler class (see
GraphSAINTSampler).


**Parameters:**
: **walk_length** ([int](https://docs.python.org/3/library/functions.html#int)) – The length of each random walk.


***class *ShaDowKHopSampler(*data: [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)*, *depth: [int](https://docs.python.org/3/library/functions.html#int)*, *num_neighbors: [int](https://docs.python.org/3/library/functions.html#int)*, *node_idx: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *replace: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, ***kwargs*)[[source]](../_modules/torch_geometric/loader/shadow.html#ShaDowKHopSampler)**
: The ShaDow $k$-hop sampler from the [“Decoupling the Depth and
Scope of Graph Neural Networks”](https://arxiv.org/abs/2201.07858) paper.
Given a graph in a `data` object, the sampler will create shallow,
localized subgraphs.
A deep GNN on this local graph then smooths the informative local signals.


> **Note:** For an example of using ShaDowKHopSampler, see
[examples/shadow.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/shadow.py).


**Parameters:**
: - **data** ([torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)) – The graph data object.
- **depth** ([int](https://docs.python.org/3/library/functions.html#int)) – The depth/number of hops of the localized subgraph.
- **num_neighbors** ([int](https://docs.python.org/3/library/functions.html#int)) – The number of neighbors to sample for each node in
each hop.
- **node_idx** (*LongTensor** or **BoolTensor**, **optional*) – The nodes that should be
considered for creating mini-batches.
If set to [None](https://docs.python.org/3/library/constants.html#None), all nodes will be
considered.
- **replace** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will sample neighbors
with replacement. (default: [False](https://docs.python.org/3/library/constants.html#False))
- ****kwargs** (*optional*) – Additional arguments of
[torch.utils.data.DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader), such as `batch_size` or
`num_workers`.


***class *RandomNodeLoader(*data: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)]*, *num_parts: [int](https://docs.python.org/3/library/functions.html#int)*, ***kwargs*)[[source]](../_modules/torch_geometric/loader/random_node_loader.html#RandomNodeLoader)**
: A data loader that randomly samples nodes within a graph and returns
their induced subgraph.


> **Note:** For an example of using
RandomNodeLoader, see
[examples/ogbn_proteins_deepgcn.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_proteins_deepgcn.py).


**Parameters:**
: - **data** ([torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)* or *[torch_geometric.data.HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)) – The [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) or
[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData) graph object.
- **num_parts** ([int](https://docs.python.org/3/library/functions.html#int)) – The number of partitions.
- ****kwargs** (*optional*) – Additional arguments of
[torch.utils.data.DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader), such as `num_workers`.


***class *ZipLoader(*loaders: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[NodeLoader], [List](https://docs.python.org/3/library/typing.html#typing.List)[LinkLoader]]*, *filter_per_worker: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[bool](https://docs.python.org/3/library/functions.html#bool)] = None*, ***kwargs*)[[source]](../_modules/torch_geometric/loader/zip_loader.html#ZipLoader)**
: A loader that returns a tuple of data objects by sampling from multiple
NodeLoader or LinkLoader instances.


**Parameters:**
: - **loaders** (*List**[*NodeLoader*] or **List**[*LinkLoader*]*) – The loader instances.
- **filter_per_worker** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will filter
the returned data in each worker’s subprocess.
If set to [False](https://docs.python.org/3/library/constants.html#False), will filter the returned data in the main
process.
If set to [None](https://docs.python.org/3/library/constants.html#None), will automatically infer the decision based
on whether data partially lives on the GPU
(`filter_per_worker=True`) or entirely on the CPU
(`filter_per_worker=False`).
There exists different trade-offs for setting this option.
Specifically, setting this option to [True](https://docs.python.org/3/library/constants.html#True) for in-memory
datasets will move all features to shared memory, which may result
in too many open file handles. (default: [None](https://docs.python.org/3/library/constants.html#None))
- ****kwargs** (*optional*) – Additional arguments of
[torch.utils.data.DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader), such as `batch_size`,
`shuffle`, `drop_last` or `num_workers`.


***class *DataListLoader(*dataset: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Dataset](../generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset), [List](https://docs.python.org/3/library/typing.html#typing.List)[BaseData]]*, *batch_size: [int](https://docs.python.org/3/library/functions.html#int) = 1*, *shuffle: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, ***kwargs*)[[source]](../_modules/torch_geometric/loader/data_list_loader.html#DataListLoader)**
: A data loader which batches data objects from a
`torch_geometric.data.dataset` to a Python list.
Data objects can be either of type [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) or
[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData).


> **Note:** This data loader should be used for multi-GPU support via
`torch_geometric.nn.DataParallel`.


**Parameters:**
: - **dataset** ([Dataset](../generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset)) – The dataset from which to load the data.
- **batch_size** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – How many samples per batch to load.
(default: `1`)
- **shuffle** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), the data will be
reshuffled at every epoch. (default: [False](https://docs.python.org/3/library/constants.html#False))
- ****kwargs** (*optional*) – Additional arguments of
[torch.utils.data.DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader), such as `drop_last` or
`num_workers`.


***class *DenseDataLoader(*dataset: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Dataset](../generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)]]*, *batch_size: [int](https://docs.python.org/3/library/functions.html#int) = 1*, *shuffle: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, ***kwargs*)[[source]](../_modules/torch_geometric/loader/dense_data_loader.html#DenseDataLoader)**
: A data loader which batches data objects from a
`torch_geometric.data.dataset` to a
[torch_geometric.data.Batch](../generated/torch_geometric.data.Batch.html#torch_geometric.data.Batch) object by stacking all attributes in a
new dimension.


> **Note:** To make use of this data loader, all graph attributes in the dataset
need to have the same shape.
In particular, this data loader should only be used when working with
*dense* adjacency matrices.


**Parameters:**
: - **dataset** ([Dataset](../generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset)) – The dataset from which to load the data.
- **batch_size** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – How many samples per batch to load.
(default: `1`)
- **shuffle** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), the data will be
reshuffled at every epoch. (default: [False](https://docs.python.org/3/library/constants.html#False))
- ****kwargs** (*optional*) – Additional arguments of
[torch.utils.data.DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader), such as `drop_last` or
`num_workers`.


***class *TemporalDataLoader(*data: [TemporalData](../generated/torch_geometric.data.TemporalData.html#torch_geometric.data.TemporalData)*, *batch_size: [int](https://docs.python.org/3/library/functions.html#int) = 1*, *neg_sampling_ratio: [float](https://docs.python.org/3/library/functions.html#float) = 0.0*, ***kwargs*)[[source]](../_modules/torch_geometric/loader/temporal_dataloader.html#TemporalDataLoader)**
: A data loader which merges successive events of a
[torch_geometric.data.TemporalData](../generated/torch_geometric.data.TemporalData.html#torch_geometric.data.TemporalData) to a mini-batch.


**Parameters:**
: - **data** ([TemporalData](../generated/torch_geometric.data.TemporalData.html#torch_geometric.data.TemporalData)) – The [TemporalData](../generated/torch_geometric.data.TemporalData.html#torch_geometric.data.TemporalData)
from which to load the data.
- **batch_size** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – How many samples per batch to load.
(default: `1`)
- **neg_sampling_ratio** ([float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – The ratio of sampled negative
destination nodes to the number of positive destination nodes.
(default: `0.0`)
- ****kwargs** (*optional*) – Additional arguments of
[torch.utils.data.DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader).


***class *NeighborSampler(*edge_index: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), SparseTensor]*, *sizes: [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]*, *node_idx: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *return_e_id: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, *transform: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)] = None*, ***kwargs*)[[source]](../_modules/torch_geometric/loader/neighbor_sampler.html#NeighborSampler)**
: The neighbor sampler from the [“Inductive Representation Learning on
Large Graphs”](https://arxiv.org/abs/1706.02216) paper, which allows
for mini-batch training of GNNs on large-scale graphs where full-batch
training is not feasible.


Given a GNN with $L$ layers and a specific mini-batch of nodes
`node_idx` for which we want to compute embeddings, this module
iteratively samples neighbors and constructs bipartite graphs that simulate
the actual computation flow of GNNs.


More specifically, `sizes` denotes how much neighbors we want to
sample for each node in each layer.
This module then takes in these `sizes` and iteratively samples
`sizes[l]` for each node involved in layer `l`.
In the next layer, sampling is repeated for the union of nodes that were
already encountered.
The actual computation graphs are then returned in reverse-mode, meaning
that we pass messages from a larger set of nodes to a smaller one, until we
reach the nodes for which we originally wanted to compute embeddings.


Hence, an item returned by NeighborSampler holds the current
`batch_size`, the IDs `n_id` of all nodes involved in the
computation, and a list of bipartite graph objects via the tuple
`(edge_index, e_id, size)`, where `edge_index` represents the
bipartite edges between source and target nodes, `e_id` denotes the
IDs of original edges in the full graph, and `size` holds the shape
of the bipartite graph.
For each bipartite graph, target nodes are also included at the beginning
of the list of source nodes so that one can easily apply skip-connections
or add self-loops.


> **Warning:** NeighborSampler is deprecated and will
be removed in a future release.
Use torch_geometric.loader.NeighborLoader instead.


> **Note:** For an example of using NeighborSampler, see
[examples/reddit.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py) or
[examples/ogbn_train.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_train.py).


**Parameters:**
: - **edge_index** (*Tensor** or **SparseTensor*) – A `torch.LongTensor` or a
`torch_sparse.SparseTensor` that defines the underlying
graph connectivity/message passing flow.
`edge_index` holds the indices of a (sparse) symmetric
adjacency matrix.
If `edge_index` is of type `torch.LongTensor`, its shape
must be defined as `[2, num_edges]`, where messages from nodes
`edge_index[0]` are sent to nodes in `edge_index[1]`
(in case `flow="source_to_target"`).
If `edge_index` is of type `torch_sparse.SparseTensor`,
its sparse indices `(row, col)` should relate to
`row = edge_index[1]` and `col = edge_index[0]`.
The major difference between both formats is that we need to input
the *transposed* sparse adjacency matrix.
- **sizes** (*[*[int](https://docs.python.org/3/library/functions.html#int)*]*) – The number of neighbors to sample for each node in each
layer. If set to `sizes[l] = -1`, all neighbors are included
in layer `l`.
- **node_idx** (*LongTensor**, **optional*) – The nodes that should be considered
for creating mini-batches. If set to [None](https://docs.python.org/3/library/constants.html#None), all nodes will be
considered.
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes in the graph.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **return_e_id** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [False](https://docs.python.org/3/library/constants.html#False), will not return
original edge indices of sampled edges. This is only useful in case
when operating on graphs without edge features to save memory.
(default: [True](https://docs.python.org/3/library/constants.html#True))
- **transform** (*callable**, **optional*) – A function/transform that takes in
a sampled mini-batch and returns a transformed version.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- ****kwargs** (*optional*) – Additional arguments of
[torch.utils.data.DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader), such as `batch_size`,
`shuffle`, `drop_last` or `num_workers`.


***class *ImbalancedSampler(*dataset: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Dataset](../generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset), [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *input_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *num_samples: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*)[[source]](../_modules/torch_geometric/loader/imbalanced_sampler.html#ImbalancedSampler)**
: A weighted random sampler that randomly samples elements according to
class distribution.
As such, it will either remove samples from the majority class
(under-sampling) or add more examples from the minority class
(over-sampling).


**Graph-level sampling:**


```
from torch_geometric.loader import DataLoader, ImbalancedSampler

sampler = ImbalancedSampler(dataset)
loader = DataLoader(dataset, batch_size=64, sampler=sampler, ...)
```


**Node-level sampling:**


```
from torch_geometric.loader import NeighborLoader, ImbalancedSampler

sampler = ImbalancedSampler(data, input_nodes=data.train_mask)
loader = NeighborLoader(data, input_nodes=data.train_mask,
                        batch_size=64, num_neighbors=[-1, -1],
                        sampler=sampler, ...)
```


You can also pass in the class labels directly as a [torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor):


```
from torch_geometric.loader import NeighborLoader, ImbalancedSampler

sampler = ImbalancedSampler(data.y)
loader = NeighborLoader(data, input_nodes=data.train_mask,
                        batch_size=64, num_neighbors=[-1, -1],
                        sampler=sampler, ...)
```


**Parameters:**
: - **dataset** ([Dataset](../generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset)* or *[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)* or **Tensor*) – The dataset or class distribution
from which to sample the data, given either as a
[Dataset](../generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset),
[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), or [torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)
object.
- **input_nodes** (*Tensor**, **optional*) – The indices of nodes that are used by
the corresponding loader, *e.g.*, by
NeighborLoader.
If set to [None](https://docs.python.org/3/library/constants.html#None), all nodes will be considered.
This argument should only be set for node-level loaders and does
not have any effect when operating on a set of graphs as given by
[Dataset](../generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset). (default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_samples** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of samples to draw for a single
epoch. If set to [None](https://docs.python.org/3/library/constants.html#None), will sample as much elements as there
exists in the underlying data. (default: [None](https://docs.python.org/3/library/constants.html#None))


***class *DynamicBatchSampler(*dataset: [Dataset](../generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset)*, *max_num: [int](https://docs.python.org/3/library/functions.html#int)*, *mode: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'node'*, *shuffle: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *skip_too_big: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *num_steps: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*)[[source]](../_modules/torch_geometric/loader/dynamic_batch_sampler.html#DynamicBatchSampler)**
: Dynamically adds samples to a mini-batch up to a maximum size (either
based on number of nodes or number of edges). When data samples have a
wide range in sizes, specifying a mini-batch size in terms of number of
samples is not ideal and can cause CUDA OOM errors.


Within the DynamicBatchSampler, the number of steps per epoch is
ambiguous, depending on the order of the samples. By default the
`__len__()` will be undefined. This is fine for most cases but
progress bars will be infinite. Alternatively, `num_steps` can be
supplied to cap the number of mini-batches produced by the sampler.


```
from torch_geometric.loader import DataLoader, DynamicBatchSampler

sampler = DynamicBatchSampler(dataset, max_num=10000, mode="node")
loader = DataLoader(dataset, batch_sampler=sampler, ...)
```


**Parameters:**
: - **dataset** ([Dataset](../generated/torch_geometric.data.Dataset.html#torch_geometric.data.Dataset)) – Dataset to sample from.
- **max_num** ([int](https://docs.python.org/3/library/functions.html#int)) – Size of mini-batch to aim for in number of nodes or
edges.
- **mode** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – `"node"` or `"edge"` to measure
batch size. (default: `"node"`)
- **shuffle** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will have the data
reshuffled at every epoch. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **skip_too_big** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), skip samples
which cannot fit in a batch by itself. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **num_steps** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of mini-batches to draw for a
single epoch. If set to [None](https://docs.python.org/3/library/constants.html#None), will iterate through all the
underlying examples, but `__len__()` will be [None](https://docs.python.org/3/library/constants.html#None) since
it is ambiguous. (default: [None](https://docs.python.org/3/library/constants.html#None))


***class *PrefetchLoader(*loader: [DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader)*, *device: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[device](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.device)] = None*)[[source]](../_modules/torch_geometric/loader/prefetch.html#PrefetchLoader)**
: A GPU prefetcher class for asynchronously transferring data of a
[torch.utils.data.DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader) from host memory to device memory.


**Parameters:**
: - **loader** ([torch.utils.data.DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader)) – The data loader.
- **device** ([torch.device](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.device)*, **optional*) – The device to load the data to.
(default: [None](https://docs.python.org/3/library/constants.html#None))


***class *CachedLoader(*loader: [DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader)*, *device: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[device](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.device)] = None*, *transform: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)] = None*)[[source]](../_modules/torch_geometric/loader/cache.html#CachedLoader)**
: A loader to cache mini-batch outputs, e.g., obtained during
NeighborLoader iterations.


**Parameters:**
: - **loader** ([torch.utils.data.DataLoader](https://docs.pytorch.org/docs/main/data.html#torch.utils.data.DataLoader)) – The data loader.
- **device** ([torch.device](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.device)*, **optional*) – The device to load the data to.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **transform** (*callable**, **optional*) – A function/transform that takes in
a sampled mini-batch and returns a transformed version.
(default: [None](https://docs.python.org/3/library/constants.html#None))


**clear()[[source]](../_modules/torch_geometric/loader/cache.html#CachedLoader.clear)**
: Clears the cache.


***class *AffinityMixin[[source]](../_modules/torch_geometric/loader/mixin.html#AffinityMixin)**
: A context manager to enable CPU affinity for data loader workers
(only used when running on CPU devices).


Affinitization places data loader workers threads on specific CPU cores.
In effect, it allows for more efficient local memory allocation and reduces
remote memory calls.
Every time a process or thread moves from one core to another, registers
and caches need to be flushed and reloaded.
This can become very costly if it happens often, and our threads may also
no longer be close to their data, or be able to share data in a cache.


See [here](https://pytorch-geometric.readthedocs.io/en/latest/advanced/cpu_affinity.html) for the accompanying tutorial.


> **Warning:** To correctly affinitize compute threads (*i.e.* with
`KMP_AFFINITY`), please make sure that you exclude
`loader_cores` from the list of cores available for the main
process.
This will cause core oversubsription and exacerbate performance.


```
loader = NeigborLoader(data, num_workers=3)
with loader.enable_cpu_affinity(loader_cores=[0, 1, 2]):
    for batch in loader:
        pass
```


**enable_cpu_affinity(*loader_cores: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]], [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]] = None*) → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torch_geometric/loader/mixin.html#AffinityMixin.enable_cpu_affinity)**
: Enables CPU affinity.


**Parameters:**
: **loader_cores** (*[*[int](https://docs.python.org/3/library/functions.html#int)*]**, **optional*) – List of CPU cores to which data
loader workers should affinitize to.
By default, it will affinitize to `numa0` cores.
If used with `"spawn"` multiprocessing context, it will
automatically enable multithreading and use multiple cores
per each worker.

**Return type:**
: [None](https://docs.python.org/3/library/constants.html#None)


