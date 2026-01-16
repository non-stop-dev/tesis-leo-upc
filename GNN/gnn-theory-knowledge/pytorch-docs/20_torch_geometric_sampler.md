| BaseSampler | An abstract base class that initializes a graph sampler and providessample_from_nodes()andsample_from_edges()routines. |
| --- | --- |
| NodeSamplerInput | The sampling input ofsample_from_nodes(). |
| EdgeSamplerInput | The sampling input ofsample_from_edges(). |
| SamplerOutput | The sampling output of aBaseSampleron homogeneous graphs. |
| HeteroSamplerOutput | The sampling output of aBaseSampleron heterogeneous graphs. |
| NumNeighbors | The number of neighbors to sample in a homogeneous or heterogeneous graph. |
| NegativeSampling | The negative sampling configuration of aBaseSamplerwhen callingsample_from_edges(). |
| NeighborSampler | An implementation of an in-memory (heterogeneous) neighbor sampler used byNeighborLoader. |
| BidirectionalNeighborSampler | A sampler that allows for both upstream and downstream sampling. |
| HGTSampler | An implementation of an in-memory heterogeneous layer-wise sampler user byHGTLoader. |


***class *BaseSampler[[source]](../_modules/torch_geometric/sampler/base.html#BaseSampler)**
: An abstract base class that initializes a graph sampler and provides
sample_from_nodes() and sample_from_edges() routines.


> **Note:** Any data stored in the sampler will be *replicated* across data loading
workers that use the sampler since each data loading worker holds its
own instance of a sampler.
As such, it is recommended to limit the amount of information stored in
the sampler.


***abstract *sample_from_nodes(*index: NodeSamplerInput*, ***kwargs*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[HeteroSamplerOutput, SamplerOutput][[source]](../_modules/torch_geometric/sampler/base.html#BaseSampler.sample_from_nodes)**
: Performs sampling from the nodes specified in `index`,
returning a sampled subgraph in the specified output format.


The `index` is a tuple holding the following information:


1. The example indices of the seed nodes
2. The node indices to start sampling from
3. The timestamps of the given seed nodes (optional)


**Parameters:**
: - **index** (NodeSamplerInput) – The node sampler input object.
- ****kwargs** (*optional*) – Additional keyword arguments.

**Return type:**
: `Union`[HeteroSamplerOutput, SamplerOutput]


***abstract *sample_from_edges(*index: EdgeSamplerInput*, *neg_sampling: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[NegativeSampling] = None*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[HeteroSamplerOutput, SamplerOutput][[source]](../_modules/torch_geometric/sampler/base.html#BaseSampler.sample_from_edges)**
: Performs sampling from the edges specified in `index`,
returning a sampled subgraph in the specified output format.


The `index` is a tuple holding the following information:


1. The example indices of the seed links
2. The source node indices to start sampling from
3. The destination node indices to start sampling from
4. The labels of the seed links (optional)
5. The timestamps of the given seed nodes (optional)


**Parameters:**
: - **index** (EdgeSamplerInput) – The edge sampler input object.
- **neg_sampling** (NegativeSampling*, **optional*) – The negative sampling
configuration. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: `Union`[HeteroSamplerOutput, SamplerOutput]


***property *edge_permutation*: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [None](https://docs.python.org/3/library/constants.html#None), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]]***
: If the sampler performs any modification of edge ordering in the
original graph, this function is expected to return the permutation
tensor that defines the permutation from the edges in the original
graph and the edges used in the sampler. If no such permutation was
applied, [None](https://docs.python.org/3/library/constants.html#None) is returned. For heterogeneous graphs, the
expected return type is a permutation tensor for each edge type.


**Return type:**
: `Union`[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [None](https://docs.python.org/3/library/constants.html#None), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]]


Graph sampler package.


***class *NodeSamplerInput(*input_id: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *node: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *time: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *input_type: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*)[[source]](../_modules/torch_geometric/sampler/base.html#NodeSamplerInput)**
: The sampling input of
sample_from_nodes().


**Parameters:**
: - **input_id** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The indices of the data loader input
of the current mini-batch.
- **node** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The indices of seed nodes to start sampling from.
- **time** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The timestamp for the seed nodes.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **input_type** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The input node type (in case of sampling in
a heterogeneous graph). (default: [None](https://docs.python.org/3/library/constants.html#None))


***class *EdgeSamplerInput(*input_id: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *row: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *col: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *label: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *time: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *input_type: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*)[[source]](../_modules/torch_geometric/sampler/base.html#EdgeSamplerInput)**
: The sampling input of
sample_from_edges().


**Parameters:**
: - **input_id** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The indices of the data loader input
of the current mini-batch.
- **row** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The source node indices of seed links to start
sampling from.
- **col** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The destination node indices of seed links to start
sampling from.
- **label** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The label for the seed links.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **time** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The timestamp for the seed links.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **input_type** (*Tuple**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*]**, **optional*) – The input edge type (in
case of sampling in a heterogeneous graph). (default: [None](https://docs.python.org/3/library/constants.html#None))


***class *SamplerOutput(*node: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *row: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *col: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *batch: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *num_sampled_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]] = None*, *num_sampled_edges: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]] = None*, *orig_row: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *orig_col: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *metadata: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*, *_seed_node: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*)[[source]](../_modules/torch_geometric/sampler/base.html#SamplerOutput)**
: The sampling output of a BaseSampler
on homogeneous graphs.


**Parameters:**
: - **node** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The sampled nodes in the original graph.
- **row** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The source node indices of the sampled subgraph.
Indices must be re-indexed to `{ 0, ..., num_nodes - 1 }`
corresponding to the nodes in the `node` tensor.
- **col** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The destination node indices of the sampled
subgraph.
Indices must be re-indexed to `{ 0, ..., num_nodes - 1 }`
corresponding to the nodes in the `node` tensor.
- **edge** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The sampled edges in the original graph.
This tensor is used to obtain edge features from the original
graph. If no edge attributes are present, it may be omitted.
- **batch** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The vector to identify the seed node
for each sampled node. Can be present in case of disjoint subgraph
sampling per seed node. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_sampled_nodes** (*List**[*[int](https://docs.python.org/3/library/functions.html#int)*]**, **optional*) – The number of sampled nodes
per hop. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_sampled_edges** (*List**[*[int](https://docs.python.org/3/library/functions.html#int)*]**, **optional*) – The number of sampled edges
per hop. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **orig_row** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The original source node indices
returned by the sampler.
Filled in case to_bidirectional() is called with the
`keep_orig_edges` option. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **orig_col** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The original destination node
indices indices returned by the sampler.
Filled in case to_bidirectional() is called with the
`keep_orig_edges` option. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **metadata** ([Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)], default: `None`) – (Any, optional): Additional metadata information.
(default: [None](https://docs.python.org/3/library/constants.html#None))


**to_bidirectional(*keep_orig_edges: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → SamplerOutput[[source]](../_modules/torch_geometric/sampler/base.html#SamplerOutput.to_bidirectional)**
: Converts the sampled subgraph into a bidirectional variant, in
which all sampled edges are guaranteed to be bidirectional.


**Parameters:**
: **keep_orig_edges** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If specified, directional edges
are still maintained. (default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: SamplerOutput


***classmethod *collate(*outputs: [List](https://docs.python.org/3/library/typing.html#typing.List)[SamplerOutput]*, *replace: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → SamplerOutput[[source]](../_modules/torch_geometric/sampler/base.html#SamplerOutput.collate)**
: Collate a list of SamplerOutput
objects into a single SamplerOutput
object. Requires that they all have the same fields.


**Return type:**
: SamplerOutput


**merge_with(*other: SamplerOutput*, *replace: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → SamplerOutput[[source]](../_modules/torch_geometric/sampler/base.html#SamplerOutput.merge_with)**
: Merges two SamplerOutputs.
If replace is True, self’s nodes and edges take precedence.


**Return type:**
: SamplerOutput


***class *HeteroSamplerOutput(*node: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *row: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *col: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *edge: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]*, *batch: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]] = None*, *num_sampled_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]] = None*, *num_sampled_edges: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]] = None*, *orig_row: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]] = None*, *orig_col: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]] = None*, *metadata: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*)[[source]](../_modules/torch_geometric/sampler/base.html#HeteroSamplerOutput)**
: The sampling output of a BaseSampler
on heterogeneous graphs.


**Parameters:**
: - **node** (*Dict**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]*) – The sampled nodes in the original graph
for each node type.
- **row** (*Dict**[**Tuple**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*]**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]*) – The source node indices
of the sampled subgraph for each edge type.
Indices must be re-indexed to `{ 0, ..., num_nodes - 1 }`
corresponding to the nodes in the `node` tensor of the source
node type.
- **col** (*Dict**[**Tuple**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*]**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]*) – The destination node
indices of the sampled subgraph for each edge type.
Indices must be re-indexed to `{ 0, ..., num_nodes - 1 }`
corresponding to the nodes in the `node` tensor of the
destination node type.
- **edge** (*Dict**[**Tuple**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*]**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]**, **optional*) – The sampled
edges in the original graph for each edge type.
This tensor is used to obtain edge features from the original
graph. If no edge attributes are present, it may be omitted.
- **batch** (*Dict**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]**, **optional*) – The vector to identify the
seed node for each sampled node for each node type. Can be present
in case of disjoint subgraph sampling per seed node.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_sampled_nodes** (*Dict**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, **List**[*[int](https://docs.python.org/3/library/functions.html#int)*]**]**, **optional*) – The number of
sampled nodes for each node type and each layer.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_sampled_edges** (*Dict**[**EdgeType**, **List**[*[int](https://docs.python.org/3/library/functions.html#int)*]**]**, **optional*) – The number of
sampled edges for each edge type and each layer.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **orig_row** (*Dict**[**EdgeType**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]**, **optional*) – The original source
node indices returned by the sampler.
Filled in case to_bidirectional() is called with the
`keep_orig_edges` option. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **orig_col** (*Dict**[**EdgeType**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]**, **optional*) – The original
destination node indices returned by the sampler.
Filled in case to_bidirectional() is called with the
`keep_orig_edges` option. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **metadata** ([Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)], default: `None`) – (Any, optional): Additional metadata information.
(default: [None](https://docs.python.org/3/library/constants.html#None))


**to_bidirectional(*keep_orig_edges: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → SamplerOutput[[source]](../_modules/torch_geometric/sampler/base.html#HeteroSamplerOutput.to_bidirectional)**
: Converts the sampled subgraph into a bidirectional variant, in
which all sampled edges are guaranteed to be bidirectional.


**Parameters:**
: **keep_orig_edges** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If specified, directional edges
are still maintained. (default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: SamplerOutput


***classmethod *collate(*outputs: [List](https://docs.python.org/3/library/typing.html#typing.List)[HeteroSamplerOutput]*, *replace: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → HeteroSamplerOutput[[source]](../_modules/torch_geometric/sampler/base.html#HeteroSamplerOutput.collate)**
: Collate a list of
`HeteroSamplerOutput` object.
Requires that they all have the same fields.


**Return type:**
: HeteroSamplerOutput


**merge_with(*other: HeteroSamplerOutput*, *replace: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → HeteroSamplerOutput[[source]](../_modules/torch_geometric/sampler/base.html#HeteroSamplerOutput.merge_with)**
: Merges two HeteroSamplerOutputs.
If replace is True, self’s nodes and edges take precedence.


**Return type:**
: HeteroSamplerOutput


***class *NumNeighbors(*values: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]]*, *default: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]] = None*)[[source]](../_modules/torch_geometric/sampler/base.html#NumNeighbors)**
: The number of neighbors to sample in a homogeneous or heterogeneous
graph. In heterogeneous graphs, may also take in a dictionary denoting
the amount of neighbors to sample for individual edge types.


**Parameters:**
: - **values** (*List**[*[int](https://docs.python.org/3/library/functions.html#int)*] or **Dict**[**Tuple**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*]**, **List**[*[int](https://docs.python.org/3/library/functions.html#int)*]**]*) – The
number of neighbors to sample.
If an entry is set to `-1`, all neighbors will be included.
In heterogeneous graphs, may also take in a dictionary denoting
the amount of neighbors to sample for individual edge types.
- **default** (*List**[*[int](https://docs.python.org/3/library/functions.html#int)*]**, **optional*) – The default number of neighbors for edge
types not specified in `values`. (default: [None](https://docs.python.org/3/library/constants.html#None))


**get_values(*edge_types: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]]] = None*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]][[source]](../_modules/torch_geometric/sampler/base.html#NumNeighbors.get_values)**
: Returns the number of neighbors.


**Parameters:**
: **edge_types** (*List**[**Tuple**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*]**]**, **optional*) – The edge types
to generate the number of neighbors for. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: `Union`[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]]


**get_mapped_values(*edge_types: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]]] = None*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]][[source]](../_modules/torch_geometric/sampler/base.html#NumNeighbors.get_mapped_values)**
: Returns the number of neighbors.
For heterogeneous graphs, a dictionary is returned in which edge type
tuples are converted to strings.


**Parameters:**
: **edge_types** (*List**[**Tuple**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*, *[str](https://docs.python.org/3/library/stdtypes.html#str)*]**]**, **optional*) – The edge types
to generate the number of neighbors for. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: `Union`[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]]


***property *num_hops*: [int](https://docs.python.org/3/library/functions.html#int)***
: Returns the number of hops.


**Return type:**
: [int](https://docs.python.org/3/library/functions.html#int)


***class *NegativeSampling(*mode: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[NegativeSamplingMode, [str](https://docs.python.org/3/library/stdtypes.html#str)]*, *amount: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)] = 1*, *src_weight: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *dst_weight: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*)[[source]](../_modules/torch_geometric/sampler/base.html#NegativeSampling)**
: The negative sampling configuration of a
BaseSampler when calling
sample_from_edges().


**Parameters:**
: - **mode** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – The negative sampling mode
(`"binary"` or `"triplet"`).
If set to `"binary"`, will randomly sample negative links
from the graph.
If set to `"triplet"`, will randomly sample negative
destination nodes for each positive source node.
- **amount** ([int](https://docs.python.org/3/library/functions.html#int)* or *[float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – The ratio of sampled negative edges to
the number of positive edges. (default: `1`)
- **src_weight** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – A node-level vector determining
the sampling of source nodes. Does not necessarily need to sum up
to one. If not given, negative nodes will be sampled uniformly.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **dst_weight** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – A node-level vector determining
the sampling of destination nodes. Does not necessarily need to sum
up to one. If not given, negative nodes will be sampled uniformly.
(default: [None](https://docs.python.org/3/library/constants.html#None))


**sample(*num_samples: [int](https://docs.python.org/3/library/functions.html#int)*, *endpoint: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['src', 'dst']*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/sampler/base.html#NegativeSampling.sample)**
: Generates `num_samples` negative samples.


**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


***class *NeighborSampler(*data: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[FeatureStore](../generated/torch_geometric.data.FeatureStore.html#torch_geometric.data.FeatureStore), [GraphStore](../generated/torch_geometric.data.GraphStore.html#torch_geometric.data.GraphStore)]]*, *num_neighbors: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[NumNeighbors, [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]]*, *subgraph_type: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[SubgraphType, [str](https://docs.python.org/3/library/stdtypes.html#str)] = 'directional'*, *replace: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *disjoint: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *temporal_strategy: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'uniform'*, *time_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *weight_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *is_sorted: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *share_memory: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *directed: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, *sample_direction: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['forward', 'backward'] = 'forward'*)[[source]](../_modules/torch_geometric/sampler/neighbor_sampler.html#NeighborSampler)**
: An implementation of an in-memory (heterogeneous) neighbor sampler used
by [NeighborLoader](loader.html#torch_geometric.loader.NeighborLoader).


***class *BidirectionalNeighborSampler(*data: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[FeatureStore](../generated/torch_geometric.data.FeatureStore.html#torch_geometric.data.FeatureStore), [GraphStore](../generated/torch_geometric.data.GraphStore.html#torch_geometric.data.GraphStore)]]*, *num_neighbors: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[NumNeighbors, [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]]*, *subgraph_type: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[SubgraphType, [str](https://docs.python.org/3/library/stdtypes.html#str)] = 'directional'*, *replace: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *disjoint: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *temporal_strategy: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'uniform'*, *time_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *weight_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *is_sorted: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *share_memory: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *directed: [bool](https://docs.python.org/3/library/functions.html#bool) = True*)[[source]](../_modules/torch_geometric/sampler/neighbor_sampler.html#BidirectionalNeighborSampler)**
: A sampler that allows for both upstream and downstream sampling.


***class *HGTSampler(*data: [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)*, *num_samples: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]]*, *is_sorted: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *share_memory: [bool](https://docs.python.org/3/library/functions.html#bool) = False*)[[source]](../_modules/torch_geometric/sampler/hgt_sampler.html#HGTSampler)**
: An implementation of an in-memory heterogeneous layer-wise sampler
user by [HGTLoader](loader.html#torch_geometric.loader.HGTLoader).


