| scatter | Reduces all values from thesrctensor at the indices specified in theindextensor along a given dimensiondim. |
| --- | --- |
| group_argsort | Returns the indices that sort the tensorsrcalong a given dimension in ascending order by value. |
| group_cat | Concatenates the given sequence of tensorstensorsin the given dimensiondim. |
| segment | Reduces all values in the first dimension of thesrctensor within the ranges specified in theptr. |
| index_sort | Sorts the elements of theinputstensor in ascending order. |
| cumsum | Returns the cumulative sum of elements ofx. |
| degree | Computes the (unweighted) degree of a given one-dimensional index tensor. |
| softmax | Computes a sparsely evaluated softmax. |
| lexsort | Performs an indirect stable sort using a sequence of keys. |
| sort_edge_index | Row-wise sortsedge_index. |
| coalesce | Row-wise sortsedge_indexand removes its duplicated entries. |
| is_undirected | ReturnsTrueif the graph given byedge_indexis undirected. |
| to_undirected | Converts the graph given byedge_indexto an undirected graph such that$(j,i) \in \mathcal{E}$for every edge$(i,j) \in \mathcal{E}$. |
| contains_self_loops | ReturnsTrueif the graph given byedge_indexcontains self-loops. |
| remove_self_loops | Removes every self-loop in the graph given byedge_index, so that$(i,i) \not\in \mathcal{E}$for every$i \in \mathcal{V}$. |
| segregate_self_loops | Segregates self-loops from the graph. |
| add_self_loops | Adds a self-loop$(i,i) \in \mathcal{E}$to every node$i \in \mathcal{V}$in the graph given byedge_index. |
| add_remaining_self_loops | Adds remaining self-loop$(i,i) \in \mathcal{E}$to every node$i \in \mathcal{V}$in the graph given byedge_index. |
| get_self_loop_attr | Returns the edge features or weights of self-loops$(i, i)$of every node$i \in \mathcal{V}$in the graph given byedge_index. |
| contains_isolated_nodes | ReturnsTrueif the graph given byedge_indexcontains isolated nodes. |
| remove_isolated_nodes | Removes the isolated nodes from the graph given byedge_indexwith optional edge attributesedge_attr. |
| get_num_hops | Returns the number of hops the model is aggregating information from. |
| subgraph | Returns the induced subgraph of(edge_index,edge_attr)containing the nodes insubset. |
| bipartite_subgraph | Returns the induced subgraph of the bipartite graph(edge_index,edge_attr)containing the nodes insubset. |
| k_hop_subgraph | Computes the induced subgraph ofedge_indexaround all nodes innode_idxreachable within$k$hops. |
| dropout_node | Randomly drops nodes from the adjacency matrixedge_indexwith probabilitypusing samples from a Bernoulli distribution. |
| dropout_edge | Randomly drops edges from the adjacency matrixedge_indexwith probabilitypusing samples from a Bernoulli distribution. |
| dropout_path | Drops edges from the adjacency matrixedge_indexbased on random walks. |
| dropout_adj | Randomly drops edges from the adjacency matrix(edge_index,edge_attr)with probabilitypusing samples from a Bernoulli distribution. |
| homophily | The homophily of a graph characterizes how likely nodes with the same label are near each other in a graph. |
| assortativity | The degree assortativity coefficient from the"Mixing patterns in networks"paper. |
| normalize_edge_index | Applies normalization to the edges of a graph. |
| get_laplacian | Computes the graph Laplacian of the graph given byedge_indexand optionaledge_weight. |
| get_mesh_laplacian | Computes the mesh Laplacian of a mesh given byposandface. |
| mask_select | Returns a new tensor which masks thesrctensor along the dimensiondimaccording to the boolean maskmask. |
| index_to_mask | Converts indices to a mask representation. |
| mask_to_index | Converts a mask to an index representation. |
| select | Selects the input tensor or input list according to a given index or mask vector. |
| narrow | Narrows the input tensor or input list to the specified range. |
| to_dense_batch | Given a sparse batch of node features$\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}$(with$N_i$indicating the number of nodes in graph$i$), creates a dense node feature tensor$\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}$(with$N_{\max} = \max_i^B N_i$). |
| to_dense_adj | Converts batched sparse adjacency matrices given by edge indices and edge attributes to a single dense batched adjacency matrix. |
| to_nested_tensor | Given a contiguous batch of tensors$\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}$(with$N_i$indicating the number of elements in example$i$), creates anested PyTorch tensor. |
| from_nested_tensor | Given anested PyTorch tensor, creates a contiguous batch of tensors$\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}$, and optionally a batch vector which assigns each element to a specific example. |
| dense_to_sparse | Converts a dense adjacency matrix to a sparse adjacency matrix defined by edge indices and edge attributes. |
| is_torch_sparse_tensor | ReturnsTrueif the inputsrcis atorch.sparse.Tensor(in any sparse layout). |
| is_sparse | ReturnsTrueif the inputsrcis of typetorch.sparse.Tensor(in any sparse layout) or of typetorch_sparse.SparseTensor. |
| to_torch_coo_tensor | Converts a sparse adjacency matrix defined by edge indices and edge attributes to atorch.sparse.Tensorwith layouttorch.sparse_coo. |
| to_torch_csr_tensor | Converts a sparse adjacency matrix defined by edge indices and edge attributes to atorch.sparse.Tensorwith layouttorch.sparse_csr. |
| to_torch_csc_tensor | Converts a sparse adjacency matrix defined by edge indices and edge attributes to atorch.sparse.Tensorwith layouttorch.sparse_csc. |
| to_torch_sparse_tensor | Converts a sparse adjacency matrix defined by edge indices and edge attributes to atorch.sparse.Tensorwith customlayout. |
| to_edge_index | Converts atorch.sparse.Tensoror atorch_sparse.SparseTensorto edge indices and edge attributes. |
| spmm | Matrix product of sparse matrix with dense matrix. |
| unbatch | Splitssrcaccording to abatchvector along dimensiondim. |
| unbatch_edge_index | Splits theedge_indexaccording to abatchvector. |
| one_hot | Taskes a one-dimensionalindextensor and returns a one-hot encoded representation of it with shape[*,num_classes]that has zeros everywhere except where the index of last dimension matches the corresponding value of the input tensor, in which case it will be1. |
| normalized_cut | Computes the normalized cut$\mathbf{e}_{i,j} \cdot \left( \frac{1}{\deg(i)} + \frac{1}{\deg(j)} \right)$of a weighted graph given by edge indices and edge attributes. |
| grid | Returns the edge indices of a two-dimensional grid graph with heightheightand widthwidthand its node positions. |
| geodesic_distance | Computes (normalized) geodesic distances of a mesh given byposandface. |
| to_scipy_sparse_matrix | Converts a graph given by edge indices and edge attributes to a scipy sparse matrix. |
| from_scipy_sparse_matrix | Converts a scipy sparse matrix to edge indices and edge attributes. |
| to_networkx | Converts atorch_geometric.data.Datainstance to anetworkx.Graphifto_undirectedis set toTrue, or a directednetworkx.DiGraphotherwise. |
| from_networkx | Converts anetworkx.Graphornetworkx.DiGraphto atorch_geometric.data.Datainstance. |
| to_networkit | Converts a(edge_index,edge_weight)tuple to anetworkit.Graph. |
| from_networkit | Converts anetworkit.Graphto a(edge_index,edge_weight)tuple. |
| to_trimesh | Converts atorch_geometric.data.Datainstance to atrimesh.Trimesh. |
| from_trimesh | Converts atrimesh.Trimeshto atorch_geometric.data.Datainstance. |
| to_cugraph | Converts a graph given byedge_indexand optionaledge_weightinto acugraphgraph object. |
| from_cugraph | Converts acugraphgraph object intoedge_indexand optionaledge_weighttensors. |
| to_dgl | Converts atorch_geometric.data.Dataortorch_geometric.data.HeteroDatainstance to adglgraph object. |
| from_dgl | Converts adglgraph object to atorch_geometric.data.Dataortorch_geometric.data.HeteroDatainstance. |
| from_rdmol | Converts ardkit.Chem.Molinstance to atorch_geometric.data.Datainstance. |
| to_rdmol | Converts atorch_geometric.data.Datainstance to ardkit.Chem.Molinstance. |
| from_smiles | Converts a SMILES string to atorch_geometric.data.Datainstance. |
| to_smiles | Converts atorch_geometric.data.Datainstance to a SMILES string. |
| erdos_renyi_graph | Returns theedge_indexof a random Erdos-Renyi graph. |
| stochastic_blockmodel_graph | Returns theedge_indexof a stochastic blockmodel graph. |
| barabasi_albert_graph | Returns theedge_indexof a Barabasi-Albert preferential attachment model, where a graph ofnum_nodesnodes grows by attaching new nodes withnum_edgesedges that are preferentially attached to existing nodes with high degree. |
| negative_sampling | Samples random negative edges of a graph given byedge_index. |
| batched_negative_sampling | Samples random negative edges of multiple graphs given byedge_indexandbatch. |
| structured_negative_sampling | Samples a negative edge(i,k)for every positive edge(i,j)in the graph given byedge_index, and returns it as a tuple of the form(i,j,k). |
| shuffle_node | Randomly shuffle the feature matrixxalong the first dimension. |
| mask_feature | Randomly masks feature from the feature matrixxwith probabilitypusing samples from a Bernoulli distribution. |
| add_random_edge | Randomly adds edges toedge_index. |
| tree_decomposition | The tree decomposition algorithm of molecules from the"Junction Tree Variational Autoencoder for Molecular Graph Generation"paper. |
| get_embeddings | Returns the output embeddings of allMessagePassinglayers inmodel. |
| get_embeddings_hetero | Returns the output embeddings of allMessagePassinglayers in a heterogeneousmodel, organized by edge type. |
| trim_to_layer | Trims theedge_indexrepresentation, node featuresxand edge featuresedge_attrto a minimal-sized representation for the current GNN layerlayerin directedNeighborLoaderscenarios. |
| get_ppr | Calculates the personalized PageRank (PPR) vector for all or a subset of nodes using a variant of theAndersen algorithm. |
| train_test_split_edges | Splits the edges of atorch_geometric.data.Dataobject into positive and negative train/val/test edges. |
| total_influence | Compute Jacobian‑based influence aggregates formultipleseed nodes, as introduced in the"Towards Quantifying Long-Range Interactions in Graph Machine Learning: a Large Graph Dataset and a Measurement"paper. |


Utility package.


**scatter(*src: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *dim: [int](https://docs.python.org/3/library/functions.html#int) = 0*, *dim_size: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *reduce: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'sum'*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/_scatter.html#scatter)**
: Reduces all values from the `src` tensor at the indices specified
in the `index` tensor along a given dimension `dim`. See the
[documentation](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html)  # noqa: E501
of the `torch_scatter` package for more information.


**Parameters:**
: - **src** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The source tensor.
- **index** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The index tensor.
- **dim** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The dimension along which to index.
(default: `0`)
- **dim_size** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The size of the output tensor at dimension
`dim`. If set to [None](https://docs.python.org/3/library/constants.html#None), will create a minimal-sized output
tensor according to `index.max() + 1`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **reduce** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The reduce operation (`"sum"`, `"mean"`,
`"mul"`, `"min"`, `"max"` or `"any"`). (default: `"sum"`)

**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


**group_argsort(*src: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *dim: [int](https://docs.python.org/3/library/functions.html#int) = 0*, *num_groups: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *descending: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *return_consecutive: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *stable: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/_scatter.html#group_argsort)**
: Returns the indices that sort the tensor `src` along a given
dimension in ascending order by value.
In contrast to `torch.argsort()`, sorting is performed in groups
according to the values in `index`.


**Parameters:**
: - **src** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The source tensor.
- **index** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The index tensor.
- **dim** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The dimension along which to index.
(default: `0`)
- **num_groups** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of groups.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **descending** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – Controls the sorting order (ascending or
descending). (default: [False](https://docs.python.org/3/library/constants.html#False))
- **return_consecutive** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will not
offset the output to start from `0` for each group.
(default: [False](https://docs.python.org/3/library/constants.html#False))
- **stable** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – Controls the relative order of equivalent
elements. (default: [False](https://docs.python.org/3/library/constants.html#False))


Example


```
>>> src = torch.tensor([0, 1, 5, 4, 3, 2, 6, 7, 8])
>>> index = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2, 2])
>>> group_argsort(src, index)
tensor([0, 1, 3, 2, 1, 0, 0, 1, 2])
```


**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


**group_cat(*tensors: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)], [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), ...]]*, *indices: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)], [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), ...]]*, *dim: [int](https://docs.python.org/3/library/functions.html#int) = 0*, *return_index: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]][[source]](../_modules/torch_geometric/utils/_scatter.html#group_cat)**
: Concatenates the given sequence of tensors `tensors` in the given
dimension `dim`.
Different from `torch.cat()`, values along the concatenating dimension
are grouped according to the indices defined in the `index` tensors.
All tensors must have the same shape (except in the concatenating
dimension).


**Parameters:**
: - **tensors** (*[**Tensor**]*) – Sequence of tensors.
- **indices** (*[**Tensor**]*) – Sequence of index tensors.
- **dim** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The dimension along which the tensors are
concatenated. (default: `0`)
- **return_index** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will return the
new index tensor. (default: [False](https://docs.python.org/3/library/constants.html#False))


Example


```
>>> x1 = torch.tensor([[0.2716, 0.4233],
...                    [0.3166, 0.0142],
...                    [0.2371, 0.3839],
...                    [0.4100, 0.0012]])
>>> x2 = torch.tensor([[0.3752, 0.5782],
...                    [0.7757, 0.5999]])
>>> index1 = torch.tensor([0, 0, 1, 2])
>>> index2 = torch.tensor([0, 2])
>>> scatter_concat([x1,x2], [index1, index2], dim=0)
tensor([[0.2716, 0.4233],
        [0.3166, 0.0142],
        [0.3752, 0.5782],
        [0.2371, 0.3839],
        [0.4100, 0.0012],
        [0.7757, 0.5999]])
```


**Return type:**
: `Union`[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]


**segment(*src: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *ptr: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *reduce: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'sum'*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/_segment.html#segment)**
: Reduces all values in the first dimension of the `src` tensor
within the ranges specified in the `ptr`. See the [documentation](https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_csr.html) of the `torch_scatter` package for more
information.


**Parameters:**
: - **src** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The source tensor.
- **ptr** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – A monotonically increasing pointer tensor that
refers to the boundaries of segments such that `ptr[0] = 0`
and `ptr[-1] = src.size(0)`.
- **reduce** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The reduce operation (`"sum"`,
`"mean"`, `"min"` or `"max"`).
(default: `"sum"`)

**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


**index_sort(*inputs: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *max_value: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *stable: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/_index_sort.html#index_sort)**
: Sorts the elements of the `inputs` tensor in ascending order.
It is expected that `inputs` is one-dimensional and that it only
contains positive integer values. If `max_value` is given, it can
be used by the underlying algorithm for better performance.


**Parameters:**
: - **inputs** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – A vector with positive integer values.
- **max_value** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The maximum value stored inside
`inputs`. This value can be an estimation, but needs to be
greater than or equal to the real maximum.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **stable** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – Makes the sorting routine stable, which
guarantees that the order of equivalent elements is preserved.
(default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]


**cumsum(*x: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *dim: [int](https://docs.python.org/3/library/functions.html#int) = 0*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/functions.html#cumsum)**
: Returns the cumulative sum of elements of `x`.
In contrast to `torch.cumsum()`, prepends the output with zero.


**Parameters:**
: - **x** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The input tensor.
- **dim** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The dimension to do the operation over.
(default: `0`)


Example


```
>>> x = torch.tensor([2, 4, 1])
>>> cumsum(x)
tensor([0, 2, 6, 7])
```


**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


**degree(*index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *dtype: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[dtype](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.dtype)] = None*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/_degree.html#degree)**
: Computes the (unweighted) degree of a given one-dimensional index
tensor.


**Parameters:**
: - **index** (*LongTensor*) – Index tensor.
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max_val + 1` of `index`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **dtype** ([torch.dtype](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.dtype), optional) – The desired data type of the
returned tensor.

**Return type:**
: `Tensor`


Example


```
>>> row = torch.tensor([0, 1, 0, 2, 0])
>>> degree(row, dtype=torch.long)
tensor([3, 1, 1])
```


**softmax(*src: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *index: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *ptr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *dim: [int](https://docs.python.org/3/library/functions.html#int) = 0*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/_softmax.html#softmax)**
: Computes a sparsely evaluated softmax.
Given a value tensor `src`, this function first groups the values
along the first dimension based on the indices specified in `index`,
and then proceeds to compute the softmax individually for each group.


**Parameters:**
: - **src** (*Tensor*) – The source tensor.
- **index** (*LongTensor**, **optional*) – The indices of elements for applying the
softmax. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **ptr** (*LongTensor**, **optional*) – If given, computes the softmax based on
sorted inputs in CSR representation. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max_val + 1` of `index`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **dim** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The dimension in which to normalize.
(default: `0`)

**Return type:**
: `Tensor`


Examples


```
>>> src = torch.tensor([1., 1., 1., 1.])
>>> index = torch.tensor([0, 0, 1, 2])
>>> ptr = torch.tensor([0, 2, 3, 4])
>>> softmax(src, index)
tensor([0.5000, 0.5000, 1.0000, 1.0000])
```


```
>>> softmax(src, None, ptr)
tensor([0.5000, 0.5000, 1.0000, 1.0000])
```


```
>>> src = torch.randn(4, 4)
>>> ptr = torch.tensor([0, 4])
>>> softmax(src, index, dim=-1)
tensor([[0.7404, 0.2596, 1.0000, 1.0000],
        [0.1702, 0.8298, 1.0000, 1.0000],
        [0.7607, 0.2393, 1.0000, 1.0000],
        [0.8062, 0.1938, 1.0000, 1.0000]])
```


**lexsort(*keys: [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *dim: [int](https://docs.python.org/3/library/functions.html#int) = -1*, *descending: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/_lexsort.html#lexsort)**
: Performs an indirect stable sort using a sequence of keys.


Given multiple sorting keys, returns an array of integer indices that
describe their sort order.
The last key in the sequence is used for the primary sort order, the
second-to-last key for the secondary sort order, and so on.


**Parameters:**
: - **keys** (*[*[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]*) – The $k$ different columns to be sorted.
The last key is the primary sort key.
- **dim** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The dimension to sort along. (default: `-1`)
- **descending** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – Controls the sorting order (ascending or
descending). (default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


**sort_edge_index(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [None](https://docs.python.org/3/library/constants.html#None), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)], [str](https://docs.python.org/3/library/stdtypes.html#str)] = '???'*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *sort_by_row: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]], [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]][[source]](../_modules/torch_geometric/utils/_sort_edge_index.html#sort_edge_index)**
: Row-wise sorts `edge_index`.


**Parameters:**
: - **edge_index** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The edge indices.
- **edge_attr** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)* or **List**[*[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]**, **optional*) – Edge weights
or multi-dimensional edge features.
If given as a list, will re-shuffle and remove duplicates for all
its entries. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max_val + 1` of `edge_index`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **sort_by_row** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [False](https://docs.python.org/3/library/constants.html#False), will sort
`edge_index` column-wise/by destination node.
(default: [True](https://docs.python.org/3/library/constants.html#True))

**Return type:**
: `LongTensor` if `edge_attr` is not passed, else
(`LongTensor`, `Optional[Tensor]` or `List[Tensor]]`)


> **Warning:** From PyG >= 2.3.0 onwards, this function will always return a
tuple whenever `edge_attr` is passed as an argument (even in case
it is set to [None](https://docs.python.org/3/library/constants.html#None)).


Examples


```
>>> edge_index = torch.tensor([[2, 1, 1, 0],
                        [1, 2, 0, 1]])
>>> edge_attr = torch.tensor([[1], [2], [3], [4]])
>>> sort_edge_index(edge_index)
tensor([[0, 1, 1, 2],
        [1, 0, 2, 1]])
```


```
>>> sort_edge_index(edge_index, edge_attr)
(tensor([[0, 1, 1, 2],
        [1, 0, 2, 1]]),
tensor([[4],
        [3],
        [2],
        [1]]))
```


**coalesce(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [None](https://docs.python.org/3/library/constants.html#None), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)], [str](https://docs.python.org/3/library/stdtypes.html#str)] = '???'*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *reduce: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'sum'*, *is_sorted: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *sort_by_row: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]], [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]][[source]](../_modules/torch_geometric/utils/_coalesce.html#coalesce)**
: Row-wise sorts `edge_index` and removes its duplicated entries.
Duplicate entries in `edge_attr` are merged by scattering them
together according to the given `reduce` option.


**Parameters:**
: - **edge_index** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The edge indices.
- **edge_attr** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)* or **List**[*[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]**, **optional*) – Edge weights
or multi-dimensional edge features.
If given as a list, will re-shuffle and remove duplicates for all
its entries. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max_val + 1` of `edge_index`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **reduce** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The reduce operation to use for merging edge
features (`"sum"`, `"mean"`, `"min"`, `"max"`,
`"mul"`, `"any"`). (default: `"sum"`)
- **is_sorted** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will expect
`edge_index` to be already sorted row-wise.
- **sort_by_row** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [False](https://docs.python.org/3/library/constants.html#False), will sort
`edge_index` column-wise.

**Return type:**
: `LongTensor` if `edge_attr` is not passed, else
(`LongTensor`, `Optional[Tensor]` or `List[Tensor]]`)


> **Warning:** From PyG >= 2.3.0 onwards, this function will always return a
tuple whenever `edge_attr` is passed as an argument (even in case
it is set to [None](https://docs.python.org/3/library/constants.html#None)).


Example


```
>>> edge_index = torch.tensor([[1, 1, 2, 3],
...                            [3, 3, 1, 2]])
>>> edge_attr = torch.tensor([1., 1., 1., 1.])
>>> coalesce(edge_index)
tensor([[1, 2, 3],
        [3, 1, 2]])
```


```
>>> # Sort `edge_index` column-wise
>>> coalesce(edge_index, sort_by_row=False)
tensor([[2, 3, 1],
        [1, 2, 3]])
```


```
>>> coalesce(edge_index, edge_attr)
(tensor([[1, 2, 3],
        [3, 1, 2]]),
tensor([2., 1., 1.]))
```


```
>>> # Use 'mean' operation to merge edge features
>>> coalesce(edge_index, edge_attr, reduce='mean')
(tensor([[1, 2, 3],
        [3, 1, 2]]),
tensor([1., 1., 1.]))
```


**is_undirected(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [None](https://docs.python.org/3/library/constants.html#None), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]] = None*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [bool](https://docs.python.org/3/library/functions.html#bool)[[source]](../_modules/torch_geometric/utils/undirected.html#is_undirected)**
: Returns [True](https://docs.python.org/3/library/constants.html#True) if the graph given by `edge_index` is
undirected.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **edge_attr** (*Tensor** or **List**[**Tensor**]**, **optional*) – Edge weights or multi-
dimensional edge features.
If given as a list, will check for equivalence in all its entries.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max(edge_index) + 1`. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: [bool](https://docs.python.org/3/library/functions.html#bool)


Examples


```
>>> edge_index = torch.tensor([[0, 1, 0],
...                         [1, 0, 0]])
>>> weight = torch.tensor([0, 0, 1])
>>> is_undirected(edge_index, weight)
True
```


```
>>> weight = torch.tensor([0, 1, 1])
>>> is_undirected(edge_index, weight)
False
```


**to_undirected(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [None](https://docs.python.org/3/library/constants.html#None), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)], [str](https://docs.python.org/3/library/stdtypes.html#str)] = '???'*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *reduce: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'add'*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]], [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]][[source]](../_modules/torch_geometric/utils/undirected.html#to_undirected)**
: Converts the graph given by `edge_index` to an undirected graph
such that $(j,i) \in \mathcal{E}$ for every edge $(i,j) \in
\mathcal{E}$.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **edge_attr** (*Tensor** or **List**[**Tensor**]**, **optional*) – Edge weights or multi-
dimensional edge features.
If given as a list, will remove duplicates for all its entries.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max(edge_index) + 1`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **reduce** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The reduce operation to use for merging edge
features (`"add"`, `"mean"`, `"min"`, `"max"`,
`"mul"`). (default: `"add"`)

**Return type:**
: `LongTensor` if `edge_attr` is not passed, else
(`LongTensor`, `Optional[Tensor]` or `List[Tensor]]`)


> **Warning:** From PyG >= 2.3.0 onwards, this function will always return a
tuple whenever `edge_attr` is passed as an argument (even in case
it is set to [None](https://docs.python.org/3/library/constants.html#None)).


Examples


```
>>> edge_index = torch.tensor([[0, 1, 1],
...                            [1, 0, 2]])
>>> to_undirected(edge_index)
tensor([[0, 1, 1, 2],
        [1, 0, 2, 1]])
```


```
>>> edge_index = torch.tensor([[0, 1, 1],
...                            [1, 0, 2]])
>>> edge_weight = torch.tensor([1., 1., 1.])
>>> to_undirected(edge_index, edge_weight)
(tensor([[0, 1, 1, 2],
        [1, 0, 2, 1]]),
tensor([2., 2., 1., 1.]))
```


```
>>> # Use 'mean' operation to merge edge features
>>>  to_undirected(edge_index, edge_weight, reduce='mean')
(tensor([[0, 1, 1, 2],
        [1, 0, 2, 1]]),
tensor([1., 1., 1., 1.]))
```


**contains_self_loops(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*) → [bool](https://docs.python.org/3/library/functions.html#bool)[[source]](../_modules/torch_geometric/utils/loop.html#contains_self_loops)**
: Returns [True](https://docs.python.org/3/library/constants.html#True) if the graph given by `edge_index` contains
self-loops.


**Parameters:**
: **edge_index** (*LongTensor*) – The edge indices.

**Return type:**
: [bool](https://docs.python.org/3/library/functions.html#bool)


Examples


```
>>> edge_index = torch.tensor([[0, 1, 0],
...                            [1, 0, 0]])
>>> contains_self_loops(edge_index)
True
```


```
>>> edge_index = torch.tensor([[0, 1, 1],
...                            [1, 0, 2]])
>>> contains_self_loops(edge_index)
False
```


**remove_self_loops(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]][[source]](../_modules/torch_geometric/utils/loop.html#remove_self_loops)**
: Removes every self-loop in the graph given by `edge_index`, so
that $(i,i) \not\in \mathcal{E}$ for every $i \in \mathcal{V}$.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **edge_attr** (*Tensor**, **optional*) – Edge weights or multi-dimensional
edge features. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: (`LongTensor`, `Tensor`)


Example


```
>>> edge_index = torch.tensor([[0, 1, 0],
...                            [1, 0, 0]])
>>> edge_attr = [[1, 2], [3, 4], [5, 6]]
>>> edge_attr = torch.tensor(edge_attr)
>>> remove_self_loops(edge_index, edge_attr)
(tensor([[0, 1],
        [1, 0]]),
tensor([[1, 2],
        [3, 4]]))
```


**segregate_self_loops(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]][[source]](../_modules/torch_geometric/utils/loop.html#segregate_self_loops)**
: Segregates self-loops from the graph.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **edge_attr** (*Tensor**, **optional*) – Edge weights or multi-dimensional
edge features. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: (`LongTensor`, `Tensor`, `LongTensor`,
`Tensor`)


Example


```
>>> edge_index = torch.tensor([[0, 0, 1],
...                            [0, 1, 0]])
>>> (edge_index, edge_attr,
...  loop_edge_index,
...  loop_edge_attr) = segregate_self_loops(edge_index)
>>>  loop_edge_index
tensor([[0],
        [0]])
```


**add_self_loops(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *fill_value: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[float](https://docs.python.org/3/library/functions.html#float), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)]]] = None*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]][[source]](../_modules/torch_geometric/utils/loop.html#add_self_loops)**
: Adds a self-loop $(i,i) \in \mathcal{E}$ to every node
$i \in \mathcal{V}$ in the graph given by `edge_index`.
In case the graph is weighted or has multi-dimensional edge features
(`edge_attr != None`), edge features of self-loops will be added
according to `fill_value`.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **edge_attr** (*Tensor**, **optional*) – Edge weights or multi-dimensional edge
features. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **fill_value** ([float](https://docs.python.org/3/library/functions.html#float)* or **Tensor** or *[str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The way to generate
edge features of self-loops (in case `edge_attr != None`).
If given as [float](https://docs.python.org/3/library/functions.html#float) or [torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), edge features of
self-loops will be directly given by `fill_value`.
If given as [str](https://docs.python.org/3/library/stdtypes.html#str), edge features of self-loops are computed by
aggregating all features of edges that point to the specific node,
according to a reduce operation. (`"add"`, `"mean"`,
`"min"`, `"max"`, `"mul"`). (default: `1.`)
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)* or **Tuple**[*[int](https://docs.python.org/3/library/functions.html#int)*, *[int](https://docs.python.org/3/library/functions.html#int)*]**, **optional*) – The number of nodes,
*i.e.* `max_val + 1` of `edge_index`.
If given as a tuple, then `edge_index` is interpreted as a
bipartite graph with shape `(num_src_nodes, num_dst_nodes)`.
(default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: (`LongTensor`, `Tensor`)


Examples


```
>>> edge_index = torch.tensor([[0, 1, 0],
...                            [1, 0, 0]])
>>> edge_weight = torch.tensor([0.5, 0.5, 0.5])
>>> add_self_loops(edge_index)
(tensor([[0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1]]),
None)
```


```
>>> add_self_loops(edge_index, edge_weight)
(tensor([[0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1]]),
tensor([0.5000, 0.5000, 0.5000, 1.0000, 1.0000]))
```


```
>>> # edge features of self-loops are filled by constant `2.0`
>>> add_self_loops(edge_index, edge_weight,
...                fill_value=2.)
(tensor([[0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1]]),
tensor([0.5000, 0.5000, 0.5000, 2.0000, 2.0000]))
```


```
>>> # Use 'add' operation to merge edge features for self-loops
>>> add_self_loops(edge_index, edge_weight,
...                fill_value='add')
(tensor([[0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1]]),
tensor([0.5000, 0.5000, 0.5000, 1.0000, 0.5000]))
```


**add_remaining_self_loops(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *fill_value: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[float](https://docs.python.org/3/library/functions.html#float), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]][[source]](../_modules/torch_geometric/utils/loop.html#add_remaining_self_loops)**
: Adds remaining self-loop $(i,i) \in \mathcal{E}$ to every node
$i \in \mathcal{V}$ in the graph given by `edge_index`.
In case the graph is weighted or has multi-dimensional edge features
(`edge_attr != None`), edge features of non-existing self-loops will
be added according to `fill_value`.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **edge_attr** (*Tensor**, **optional*) – Edge weights or multi-dimensional edge
features. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **fill_value** ([float](https://docs.python.org/3/library/functions.html#float)* or **Tensor** or *[str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The way to generate
edge features of self-loops (in case `edge_attr != None`).
If given as [float](https://docs.python.org/3/library/functions.html#float) or [torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), edge features of
self-loops will be directly given by `fill_value`.
If given as [str](https://docs.python.org/3/library/stdtypes.html#str), edge features of self-loops are computed by
aggregating all features of edges that point to the specific node,
according to a reduce operation. (`"add"`, `"mean"`,
`"min"`, `"max"`, `"mul"`). (default: `1.`)
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max_val + 1` of `edge_index`. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: (`LongTensor`, `Tensor`)


Example


```
>>> edge_index = torch.tensor([[0, 1],
...                            [1, 0]])
>>> edge_weight = torch.tensor([0.5, 0.5])
>>> add_remaining_self_loops(edge_index, edge_weight)
(tensor([[0, 1, 0, 1],
        [1, 0, 0, 1]]),
tensor([0.5000, 0.5000, 1.0000, 1.0000]))
```


**get_self_loop_attr(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/loop.html#get_self_loop_attr)**
: Returns the edge features or weights of self-loops
$(i, i)$ of every node $i \in \mathcal{V}$ in the
graph given by `edge_index`. Edge features of missing self-loops not
present in `edge_index` will be filled with zeros. If
`edge_attr` is not given, it will be the vector of ones.


> **Note:** This operation is analogous to getting the diagonal elements of the
dense adjacency matrix.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **edge_attr** (*Tensor**, **optional*) – Edge weights or multi-dimensional edge
features. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max_val + 1` of `edge_index`. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: `Tensor`


Examples


```
>>> edge_index = torch.tensor([[0, 1, 0],
...                            [1, 0, 0]])
>>> edge_weight = torch.tensor([0.2, 0.3, 0.5])
>>> get_self_loop_attr(edge_index, edge_weight)
tensor([0.5000, 0.0000])
```


```
>>> get_self_loop_attr(edge_index, edge_weight, num_nodes=4)
tensor([0.5000, 0.0000, 0.0000, 0.0000])
```


**contains_isolated_nodes(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [bool](https://docs.python.org/3/library/functions.html#bool)[[source]](../_modules/torch_geometric/utils/isolated.html#contains_isolated_nodes)**
: Returns [True](https://docs.python.org/3/library/constants.html#True) if the graph given by `edge_index` contains
isolated nodes.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max_val + 1` of `edge_index`. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: [bool](https://docs.python.org/3/library/functions.html#bool)


Examples


```
>>> edge_index = torch.tensor([[0, 1, 0],
...                            [1, 0, 0]])
>>> contains_isolated_nodes(edge_index)
False
```


```
>>> contains_isolated_nodes(edge_index, num_nodes=3)
True
```


**remove_isolated_nodes(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/isolated.html#remove_isolated_nodes)**
: Removes the isolated nodes from the graph given by `edge_index`
with optional edge attributes `edge_attr`.
In addition, returns a mask of shape `[num_nodes]` to manually filter
out isolated node features later on.
Self-loops are preserved for non-isolated nodes.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **edge_attr** (*Tensor**, **optional*) – Edge weights or multi-dimensional
edge features. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max_val + 1` of `edge_index`. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: (LongTensor, Tensor, BoolTensor)


Examples


```
>>> edge_index = torch.tensor([[0, 1, 0],
...                            [1, 0, 0]])
>>> edge_index, edge_attr, mask = remove_isolated_nodes(edge_index)
>>> mask # node mask (2 nodes)
tensor([True, True])
```


```
>>> edge_index, edge_attr, mask = remove_isolated_nodes(edge_index,
...                                                     num_nodes=3)
>>> mask # node mask (3 nodes)
tensor([True, True, False])
```


**get_num_hops(*model: [Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)*) → [int](https://docs.python.org/3/library/functions.html#int)[[source]](../_modules/torch_geometric/utils/_subgraph.html#get_num_hops)**
: Returns the number of hops the model is aggregating information
from.


> **Note:** This function counts the number of message passing layers as an
approximation of the total number of hops covered by the model.
Its output may not necessarily be correct in case message passing
layers perform multi-hop aggregation, *e.g.*, as in
[ChebConv](../generated/torch_geometric.nn.conv.ChebConv.html#torch_geometric.nn.conv.ChebConv).


Example


```
>>> class GNN(torch.nn.Module):
...     def __init__(self):
...         super().__init__()
...         self.conv1 = GCNConv(3, 16)
...         self.conv2 = GCNConv(16, 16)
...         self.lin = Linear(16, 2)
...
...     def forward(self, x, edge_index):
...         x = self.conv1(x, edge_index).relu()
...         x = self.conv2(x, edge_index).relu()
...         return self.lin(x)
>>> get_num_hops(GNN())
2
```


**Return type:**
: [int](https://docs.python.org/3/library/functions.html#int)


**subgraph(*subset: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]*, *edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *relabel_nodes: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, ***, *return_edge_mask: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]], [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]][[source]](../_modules/torch_geometric/utils/_subgraph.html#subgraph)**
: Returns the induced subgraph of `(edge_index, edge_attr)`
containing the nodes in `subset`.


**Parameters:**
: - **subset** (*LongTensor**, **BoolTensor** or **[*[int](https://docs.python.org/3/library/functions.html#int)*]*) – The nodes to keep.
- **edge_index** (*LongTensor*) – The edge indices.
- **edge_attr** (*Tensor**, **optional*) – Edge weights or multi-dimensional
edge features. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **relabel_nodes** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), the resulting
`edge_index` will be relabeled to hold consecutive indices
starting from zero. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max(edge_index) + 1`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **return_edge_mask** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will return
the edge mask to filter out additional edge features.
(default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: (`LongTensor`, `Tensor`)


Examples


```
>>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6],
...                            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5]])
>>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
>>> subset = torch.tensor([3, 4, 5])
>>> subgraph(subset, edge_index, edge_attr)
(tensor([[3, 4, 4, 5],
        [4, 3, 5, 4]]),
tensor([ 7.,  8.,  9., 10.]))
```


```
>>> subgraph(subset, edge_index, edge_attr, return_edge_mask=True)
(tensor([[3, 4, 4, 5],
        [4, 3, 5, 4]]),
tensor([ 7.,  8.,  9., 10.]),
tensor([False, False, False, False, False, False,  True,
        True,  True,  True,  False, False]))
```


**bipartite_subgraph(*subset: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)], [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]]*, *edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *relabel_nodes: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *size: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)]] = None*, *return_edge_mask: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]], [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)], [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]][[source]](../_modules/torch_geometric/utils/_subgraph.html#bipartite_subgraph)**
: Returns the induced subgraph of the bipartite graph
`(edge_index, edge_attr)` containing the nodes in `subset`.


**Parameters:**
: - **subset** (*Tuple**[**Tensor**, **Tensor**] or *[tuple](https://docs.python.org/3/library/stdtypes.html#tuple)*(**[*[int](https://docs.python.org/3/library/functions.html#int)*]**,**[*[int](https://docs.python.org/3/library/functions.html#int)*]**)*) – The nodes
to keep.
- **edge_index** (*LongTensor*) – The edge indices.
- **edge_attr** (*Tensor**, **optional*) – Edge weights or multi-dimensional
edge features. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **relabel_nodes** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), the resulting
`edge_index` will be relabeled to hold consecutive indices
starting from zero. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **size** ([tuple](https://docs.python.org/3/library/stdtypes.html#tuple)*, **optional*) – The number of nodes.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **return_edge_mask** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will return
the edge mask to filter out additional edge features.
(default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: (`LongTensor`, `Tensor`)


Examples


```
>>> edge_index = torch.tensor([[0, 5, 2, 3, 3, 4, 4, 3, 5, 5, 6],
...                            [0, 0, 3, 2, 0, 0, 2, 1, 2, 3, 1]])
>>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
>>> subset = (torch.tensor([2, 3, 5]), torch.tensor([2, 3]))
>>> bipartite_subgraph(subset, edge_index, edge_attr)
(tensor([[2, 3, 5, 5],
        [3, 2, 2, 3]]),
tensor([ 3,  4,  9, 10]))
```


```
>>> bipartite_subgraph(subset, edge_index, edge_attr,
...                    return_edge_mask=True)
(tensor([[2, 3, 5, 5],
        [3, 2, 2, 3]]),
tensor([ 3,  4,  9, 10]),
tensor([False, False,  True,  True, False, False, False, False,
        True,  True,  False]))
```


**k_hop_subgraph(*node_idx: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *num_hops: [int](https://docs.python.org/3/library/functions.html#int)*, *edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *relabel_nodes: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *flow: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'source_to_target'*, *directed: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/_subgraph.html#k_hop_subgraph)**
: Computes the induced subgraph of `edge_index` around all nodes in
`node_idx` reachable within $k$ hops.


The `flow` argument denotes the direction of edges for finding
$k$-hop neighbors. If set to `"source_to_target"`, then the
method will find all neighbors that point to the initial set of seed nodes
in `node_idx.`
This mimics the natural flow of message passing in Graph Neural Networks.


The method returns (1) the nodes involved in the subgraph, (2) the filtered
`edge_index` connectivity, (3) the mapping from node indices in
`node_idx` to their new location, and (4) the edge mask indicating
which edges were preserved.


**Parameters:**
: - **node_idx** (int, list, tuple or [torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The central seed
node(s).
- **num_hops** ([int](https://docs.python.org/3/library/functions.html#int)) – The number of hops $k$.
- **edge_index** (*LongTensor*) – The edge indices.
- **relabel_nodes** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), the resulting
`edge_index` will be relabeled to hold consecutive indices
starting from zero. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max_val + 1` of `edge_index`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **flow** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The flow direction of $k$-hop aggregation
(`"source_to_target"` or `"target_to_source"`).
(default: `"source_to_target"`)
- **directed** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will only include
directed edges to the seed nodes `node_idx`.
(default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: (`LongTensor`, `LongTensor`, `LongTensor`,
`BoolTensor`)


Examples


```
>>> edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
...                            [2, 2, 4, 4, 6, 6]])
```


```
>>> # Center node 6, 2-hops
>>> subset, edge_index, mapping, edge_mask = k_hop_subgraph(
...     6, 2, edge_index, relabel_nodes=True)
>>> subset
tensor([2, 3, 4, 5, 6])
>>> edge_index
tensor([[0, 1, 2, 3],
        [2, 2, 4, 4]])
>>> mapping
tensor([4])
>>> edge_mask
tensor([False, False,  True,  True,  True,  True])
>>> subset[mapping]
tensor([6])
```


```
>>> edge_index = torch.tensor([[1, 2, 4, 5],
...                            [0, 1, 5, 6]])
>>> (subset, edge_index,
...  mapping, edge_mask) = k_hop_subgraph([0, 6], 2,
...                                       edge_index,
...                                       relabel_nodes=True)
>>> subset
tensor([0, 1, 2, 4, 5, 6])
>>> edge_index
tensor([[1, 2, 3, 4],
        [0, 1, 4, 5]])
>>> mapping
tensor([0, 5])
>>> edge_mask
tensor([True, True, True, True])
>>> subset[mapping]
tensor([0, 6])
```


**dropout_node(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *p: [float](https://docs.python.org/3/library/functions.html#float) = 0.5*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *training: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, *relabel_nodes: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/dropout.html#dropout_node)**
: Randomly drops nodes from the adjacency matrix
`edge_index` with probability `p` using samples from
a Bernoulli distribution.


The method returns (1) the retained `edge_index`, (2) the edge mask
indicating which edges were retained. (3) the node mask indicating
which nodes were retained.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **p** ([float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – Dropout probability. (default: `0.5`)
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max_val + 1` of `edge_index`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **training** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [False](https://docs.python.org/3/library/constants.html#False), this operation is a
no-op. (default: [True](https://docs.python.org/3/library/constants.html#True))
- **relabel_nodes** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to True, the resulting
edge_index will be relabeled to hold consecutive indices
starting from zero.

**Return type:**
: (`LongTensor`, `BoolTensor`, `BoolTensor`)


Examples


```
>>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
...                            [1, 0, 2, 1, 3, 2]])
>>> edge_index, edge_mask, node_mask = dropout_node(edge_index)
>>> edge_index
tensor([[0, 1],
        [1, 0]])
>>> edge_mask
tensor([ True,  True, False, False, False, False])
>>> node_mask
tensor([ True,  True, False, False])
```


**dropout_edge(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *p: [float](https://docs.python.org/3/library/functions.html#float) = 0.5*, *force_undirected: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *training: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/dropout.html#dropout_edge)**
: Randomly drops edges from the adjacency matrix
`edge_index` with probability `p` using samples from
a Bernoulli distribution.


The method returns (1) the retained `edge_index`, (2) the edge mask
or index indicating which edges were retained, depending on the argument
`force_undirected`.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **p** ([float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – Dropout probability. (default: `0.5`)
- **force_undirected** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will either
drop or keep both edges of an undirected edge.
(default: [False](https://docs.python.org/3/library/constants.html#False))
- **training** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [False](https://docs.python.org/3/library/constants.html#False), this operation is a
no-op. (default: [True](https://docs.python.org/3/library/constants.html#True))

**Return type:**
: (`LongTensor`, `BoolTensor` or `LongTensor`)


Examples


```
>>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
...                            [1, 0, 2, 1, 3, 2]])
>>> edge_index, edge_mask = dropout_edge(edge_index)
>>> edge_index
tensor([[0, 1, 2, 2],
        [1, 2, 1, 3]])
>>> edge_mask # masks indicating which edges are retained
tensor([ True, False,  True,  True,  True, False])
```


```
>>> edge_index, edge_id = dropout_edge(edge_index,
...                                    force_undirected=True)
>>> edge_index
tensor([[0, 1, 2, 1, 2, 3],
        [1, 2, 3, 0, 1, 2]])
>>> edge_id # indices indicating which edges are retained
tensor([0, 2, 4, 0, 2, 4])
```


**dropout_path(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *p: [float](https://docs.python.org/3/library/functions.html#float) = 0.2*, *walks_per_node: [int](https://docs.python.org/3/library/functions.html#int) = 1*, *walk_length: [int](https://docs.python.org/3/library/functions.html#int) = 3*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *is_sorted: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *training: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/dropout.html#dropout_path)**
: Drops edges from the adjacency matrix `edge_index`
based on random walks. The source nodes to start random walks from are
sampled from `edge_index` with probability `p`, following
a Bernoulli distribution.


The method returns (1) the retained `edge_index`, (2) the edge mask
indicating which edges were retained.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **p** ([float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – Sample probability. (default: `0.2`)
- **walks_per_node** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of walks per node, same as
[Node2Vec](../generated/torch_geometric.nn.models.Node2Vec.html#torch_geometric.nn.models.Node2Vec). (default: `1`)
- **walk_length** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The walk length, same as
[Node2Vec](../generated/torch_geometric.nn.models.Node2Vec.html#torch_geometric.nn.models.Node2Vec). (default: `3`)
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max_val + 1` of `edge_index`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **is_sorted** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will expect
`edge_index` to be already sorted row-wise.
(default: [False](https://docs.python.org/3/library/constants.html#False))
- **training** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [False](https://docs.python.org/3/library/constants.html#False), this operation is a
no-op. (default: [True](https://docs.python.org/3/library/constants.html#True))

**Return type:**
: (`LongTensor`, `BoolTensor`)


Example


```
>>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
...                            [1, 0, 2, 1, 3, 2]])
>>> edge_index, edge_mask = dropout_path(edge_index)
>>> edge_index
tensor([[1, 2],
        [2, 3]])
>>> edge_mask # masks indicating which edges are retained
tensor([False, False,  True, False,  True, False])
```


**dropout_adj(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *p: [float](https://docs.python.org/3/library/functions.html#float) = 0.5*, *force_undirected: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *training: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]][[source]](../_modules/torch_geometric/utils/dropout.html#dropout_adj)**
: Randomly drops edges from the adjacency matrix
`(edge_index, edge_attr)` with probability `p` using samples from
a Bernoulli distribution.


> **Warning:** dropout_adj is deprecated and will
be removed in a future release.
Use torch_geometric.utils.dropout_edge instead.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **edge_attr** (*Tensor**, **optional*) – Edge weights or multi-dimensional
edge features. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **p** ([float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – Dropout probability. (default: `0.5`)
- **force_undirected** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will either
drop or keep both edges of an undirected edge.
(default: [False](https://docs.python.org/3/library/constants.html#False))
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max_val + 1` of `edge_index`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **training** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [False](https://docs.python.org/3/library/constants.html#False), this operation is a
no-op. (default: [True](https://docs.python.org/3/library/constants.html#True))


Examples


```
>>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
...                            [1, 0, 2, 1, 3, 2]])
>>> edge_attr = torch.tensor([1, 2, 3, 4, 5, 6])
>>> dropout_adj(edge_index, edge_attr)
(tensor([[0, 1, 2, 3],
        [1, 2, 3, 2]]),
tensor([1, 3, 5, 6]))
```


```
>>> # The returned graph is kept undirected
>>> dropout_adj(edge_index, edge_attr, force_undirected=True)
(tensor([[0, 1, 2, 1, 2, 3],
        [1, 2, 3, 0, 1, 2]]),
tensor([1, 3, 5, 1, 3, 5]))
```


**Return type:**
: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]


**homophily(*edge_index: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), SparseTensor]*, *y: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *batch: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *method: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'edge'*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[float](https://docs.python.org/3/library/functions.html#float), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/_homophily.html#homophily)**
: The homophily of a graph characterizes how likely nodes with the same
label are near each other in a graph.


There are many measures of homophily that fits this definition.
In particular:


- In the [“Beyond Homophily in Graph Neural Networks: Current Limitations
and Effective Designs”](https://arxiv.org/abs/2006.11468) paper, the
homophily is the fraction of edges in a graph which connects nodes
that have the same class label:


$$
\frac{| \{ (v,w) : (v,w) \in \mathcal{E} \wedge y_v = y_w \} | }
{|\mathcal{E}|}
$$


That measure is called the *edge homophily ratio*.
- In the [“Geom-GCN: Geometric Graph Convolutional Networks”](https://arxiv.org/abs/2002.05287) paper, edge homophily is normalized
across neighborhoods:


$$
\frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \frac{ | \{ (w,v) : w
\in \mathcal{N}(v) \wedge y_v = y_w \} |  } { |\mathcal{N}(v)| }
$$


That measure is called the *node homophily ratio*.
- In the [“Large-Scale Learning on Non-Homophilous Graphs: New Benchmarks
and Strong Simple Methods”](https://arxiv.org/abs/2110.14446) paper,
edge homophily is modified to be insensitive to the number of classes
and size of each class:


$$
\frac{1}{C-1} \sum_{k=1}^{C} \max \left(0, h_k - \frac{|\mathcal{C}_k|}
{|\mathcal{V}|} \right),
$$


where $C$ denotes the number of classes, $|\mathcal{C}_k|$
denotes the number of nodes of class $k$, and $h_k$ denotes
the edge homophily ratio of nodes of class $k$.


Thus, that measure is called the *class insensitive edge homophily
ratio*.


**Parameters:**
: - **edge_index** (*Tensor** or **SparseTensor*) – The graph connectivity.
- **y** (*Tensor*) – The labels.
- **batch** (*LongTensor**, **optional*) – Batch vector$\mathbf{b} \in {\{ 0, \ldots,B-1\}}^N$, which assigns
each node to a specific example. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **method** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The method used to calculate the homophily,
either `"edge"` (first formula), `"node"` (second
formula) or `"edge_insensitive"` (third formula).
(default: `"edge"`)


Examples


```
>>> edge_index = torch.tensor([[0, 1, 2, 3],
...                            [1, 2, 0, 4]])
>>> y = torch.tensor([0, 0, 0, 0, 1])
>>> # Edge homophily ratio
>>> homophily(edge_index, y, method='edge')
0.75
```


```
>>> # Node homophily ratio
>>> homophily(edge_index, y, method='node')
0.6000000238418579
```


```
>>> # Class insensitive edge homophily ratio
>>> homophily(edge_index, y, method='edge_insensitive')
0.19999998807907104
```


**Return type:**
: `Union`[[float](https://docs.python.org/3/library/functions.html#float), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]


**assortativity(*edge_index: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), SparseTensor]*) → [float](https://docs.python.org/3/library/functions.html#float)[[source]](../_modules/torch_geometric/utils/_assortativity.html#assortativity)**
: The degree assortativity coefficient from the
[“Mixing patterns in networks”](https://arxiv.org/abs/cond-mat/0209450) paper.
Assortativity in a network refers to the tendency of nodes to
connect with other similar nodes over dissimilar nodes.
It is computed from Pearson correlation coefficient of the node degrees.


**Parameters:**
: **edge_index** (*Tensor** or **SparseTensor*) – The graph connectivity.

**Returns:**
: [float](https://docs.python.org/3/library/functions.html#float) – The value of the degree assortativity coefficient for the input
graph $\in [-1, 1]$


Example


```
>>> edge_index = torch.tensor([[0, 1, 2, 3, 2],
...                            [1, 2, 0, 1, 3]])
>>> assortativity(edge_index)
-0.666667640209198
```


**normalize_edge_index(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *add_self_loops: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, *symmetric: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/_normalize_edge_index.html#normalize_edge_index)**
: Applies normalization to the edges of a graph.


This function can add self-loops to the graph and apply either symmetric or
asymmetric normalization based on the node degrees.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, *[int](https://docs.python.org/3/library/functions.html#int)*]**, **optional*) – The number of nodes, *i.e.*
`max_val + 1` of `edge_index`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **add_self_loops** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [False](https://docs.python.org/3/library/constants.html#False), will not add
self-loops to the input graph. (default: [True](https://docs.python.org/3/library/constants.html#True))
- **symmetric** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), symmetric
normalization ($D^{-1/2} A D^{-1/2}$) is used, otherwise
asymmetric normalization ($D^{-1} A$).

**Return type:**
: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]


**get_laplacian(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_weight: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *normalization: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *dtype: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[dtype](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.dtype)] = None*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/laplacian.html#get_laplacian)**
: Computes the graph Laplacian of the graph given by `edge_index`
and optional `edge_weight`.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **edge_weight** (*Tensor**, **optional*) – One-dimensional edge weights.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **normalization** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) –
The normalization scheme for the graph
Laplacian (default: [None](https://docs.python.org/3/library/constants.html#None)):


1. [None](https://docs.python.org/3/library/constants.html#None): No normalization
$\mathbf{L} = \mathbf{D} - \mathbf{A}$


2. `"sym"`: Symmetric normalization
$\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
\mathbf{D}^{-1/2}$


3. `"rw"`: Random-walk normalization
$\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1} \mathbf{A}$
- **dtype** ([torch.dtype](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.dtype)*, **optional*) – The desired data type of returned tensor
in case `edge_weight=None`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max_val + 1` of `edge_index`. (default: [None](https://docs.python.org/3/library/constants.html#None))


Examples


```
>>> edge_index = torch.tensor([[0, 1, 1, 2],
...                            [1, 0, 2, 1]])
>>> edge_weight = torch.tensor([1., 2., 2., 4.])
```


```
>>> # No normalization
>>> lap = get_laplacian(edge_index, edge_weight)
```


```
>>> # Symmetric normalization
>>> lap_sym = get_laplacian(edge_index, edge_weight,
                            normalization='sym')
```


```
>>> # Random-walk normalization
>>> lap_rw = get_laplacian(edge_index, edge_weight, normalization='rw')
```


**Return type:**
: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]


**get_mesh_laplacian(*pos: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *face: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *normalization: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/mesh_laplacian.html#get_mesh_laplacian)**
: Computes the mesh Laplacian of a mesh given by `pos` and
`face`.


Computation is based on the cotangent matrix defined as


$$
\begin{split}\mathbf{C}_{ij} = \begin{cases}
      \frac{\cot \angle_{ikj}~+\cot \angle_{ilj}}{2} &
      \text{if } i, j \text{ is an edge} \\
      -\sum_{j \in N(i)}{C_{ij}} &
      \text{if } i \text{ is in the diagonal} \\
      0 & \text{otherwise}
\end{cases}\end{split}
$$


Normalization depends on the mass matrix defined as


$$
\begin{split}\mathbf{M}_{ij} = \begin{cases}
      a(i) & \text{if } i \text{ is in the diagonal} \\
      0 & \text{otherwise}
\end{cases}\end{split}
$$


where $a(i)$ is obtained by joining the barycenters of the
triangles around vertex $i$.


**Parameters:**
: - **pos** (*Tensor*) – The node positions.
- **face** (*LongTensor*) – The face indices.
- **normalization** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) –
The normalization scheme for the mesh
Laplacian (default: [None](https://docs.python.org/3/library/constants.html#None)):


1. [None](https://docs.python.org/3/library/constants.html#None): No normalization
$\mathbf{L} = \mathbf{C}$


2. `"sym"`: Symmetric normalization
$\mathbf{L} = \mathbf{M}^{-1/2} \mathbf{C}\mathbf{M}^{-1/2}$


3. `"rw"`: Row-wise normalization
$\mathbf{L} = \mathbf{M}^{-1} \mathbf{C}$

**Return type:**
: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]


**mask_select(*src: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *dim: [int](https://docs.python.org/3/library/functions.html#int)*, *mask: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/mask.html#mask_select)**
: Returns a new tensor which masks the `src` tensor along the
dimension `dim` according to the boolean mask `mask`.


**Parameters:**
: - **src** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The input tensor.
- **dim** ([int](https://docs.python.org/3/library/functions.html#int)) – The dimension in which to mask.
- **mask** (*torch.BoolTensor*) – The 1-D tensor containing the binary mask to
index with.

**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


**index_to_mask(*index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *size: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/mask.html#index_to_mask)**
: Converts indices to a mask representation.


**Parameters:**
: - **index** (*Tensor*) – The indices.
- **size** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The size of the mask. If set to [None](https://docs.python.org/3/library/constants.html#None), a
minimal sized output mask is returned.


Example


```
>>> index = torch.tensor([1, 3, 5])
>>> index_to_mask(index)
tensor([False,  True, False,  True, False,  True])
```


```
>>> index_to_mask(index, size=7)
tensor([False,  True, False,  True, False,  True, False])
```


**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


**mask_to_index(*mask: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/mask.html#mask_to_index)**
: Converts a mask to an index representation.


**Parameters:**
: **mask** (*Tensor*) – The mask.


Example


```
>>> mask = torch.tensor([False, True, False])
>>> mask_to_index(mask)
tensor([1])
```


**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


**select(*src: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)], TensorFrame]*, *index_or_mask: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *dim: [int](https://docs.python.org/3/library/functions.html#int)*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)]][[source]](../_modules/torch_geometric/utils/_select.html#select)**
: Selects the input tensor or input list according to a given index or
mask vector.


**Parameters:**
: - **src** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)* or *[list](https://docs.python.org/3/library/stdtypes.html#list)) – The input tensor or list.
- **index_or_mask** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The index or mask vector.
- **dim** ([int](https://docs.python.org/3/library/functions.html#int)) – The dimension along which to select.

**Return type:**
: `Union`[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)]]


**narrow(*src: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)]]*, *dim: [int](https://docs.python.org/3/library/functions.html#int)*, *start: [int](https://docs.python.org/3/library/functions.html#int)*, *length: [int](https://docs.python.org/3/library/functions.html#int)*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)]][[source]](../_modules/torch_geometric/utils/_select.html#narrow)**
: Narrows the input tensor or input list to the specified range.


**Parameters:**
: - **src** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)* or *[list](https://docs.python.org/3/library/stdtypes.html#list)) – The input tensor or list.
- **dim** ([int](https://docs.python.org/3/library/functions.html#int)) – The dimension along which to narrow.
- **start** ([int](https://docs.python.org/3/library/functions.html#int)) – The starting dimension.
- **length** ([int](https://docs.python.org/3/library/functions.html#int)) – The distance to the ending dimension.

**Return type:**
: `Union`[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)]]


**to_dense_batch(*x: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *batch: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *fill_value: [float](https://docs.python.org/3/library/functions.html#float) = 0.0*, *max_num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *batch_size: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/_to_dense_batch.html#to_dense_batch)**
: Given a sparse batch of node features
$\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}$ (with
$N_i$ indicating the number of nodes in graph $i$), creates a
dense node feature tensor
$\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}$ (with
$N_{\max} = \max_i^B N_i$).
In addition, a mask of shape $\mathbf{M} \in \{ 0, 1 \}^{B \times
N_{\max}}$ is returned, holding information about the existence of
fake-nodes in the dense representation.


**Parameters:**
: - **x** (*Tensor*) – Node feature matrix
$\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}$.
- **batch** (*LongTensor**, **optional*) – Batch vector
$\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N$, which assigns each
node to a specific example. Must be ordered. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **fill_value** ([float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – The value for invalid entries in the
resulting dense output tensor. (default: `0`)
- **max_num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The size of the output node dimension.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **batch_size** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The batch size. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: (`Tensor`, `BoolTensor`)


Examples


```
>>> x = torch.arange(12).view(6, 2)
>>> x
tensor([[ 0,  1],
        [ 2,  3],
        [ 4,  5],
        [ 6,  7],
        [ 8,  9],
        [10, 11]])
```


```
>>> out, mask = to_dense_batch(x)
>>> mask
tensor([[True, True, True, True, True, True]])
```


```
>>> batch = torch.tensor([0, 0, 1, 2, 2, 2])
>>> out, mask = to_dense_batch(x, batch)
>>> out
tensor([[[ 0,  1],
        [ 2,  3],
        [ 0,  0]],
        [[ 4,  5],
        [ 0,  0],
        [ 0,  0]],
        [[ 6,  7],
        [ 8,  9],
        [10, 11]]])
>>> mask
tensor([[ True,  True, False],
        [ True, False, False],
        [ True,  True,  True]])
```


```
>>> out, mask = to_dense_batch(x, batch, max_num_nodes=4)
>>> out
tensor([[[ 0,  1],
        [ 2,  3],
        [ 0,  0],
        [ 0,  0]],
        [[ 4,  5],
        [ 0,  0],
        [ 0,  0],
        [ 0,  0]],
        [[ 6,  7],
        [ 8,  9],
        [10, 11],
        [ 0,  0]]])
```


```
>>> mask
tensor([[ True,  True, False, False],
        [ True, False, False, False],
        [ True,  True,  True, False]])
```


**to_dense_adj(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *batch: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *max_num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *batch_size: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/_to_dense_adj.html#to_dense_adj)**
: Converts batched sparse adjacency matrices given by edge indices and
edge attributes to a single dense batched adjacency matrix.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **batch** (*LongTensor**, **optional*) – Batch vector
$\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N$, which assigns each
node to a specific example. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **edge_attr** (*Tensor**, **optional*) – Edge weights or multi-dimensional edge
features.
If `edge_index` contains duplicated edges, the dense adjacency
matrix output holds the summed up entries of `edge_attr` for
duplicated edges. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **max_num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The size of the output node dimension.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **batch_size** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The batch size. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: `Tensor`


Examples


```
>>> edge_index = torch.tensor([[0, 0, 1, 2, 3],
...                            [0, 1, 0, 3, 0]])
>>> batch = torch.tensor([0, 0, 1, 1])
>>> to_dense_adj(edge_index, batch)
tensor([[[1., 1.],
        [1., 0.]],
        [[0., 1.],
        [1., 0.]]])
```


```
>>> to_dense_adj(edge_index, batch, max_num_nodes=4)
tensor([[[1., 1., 0., 0.],
        [1., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]],
        [[0., 1., 0., 0.],
        [1., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]]])
```


```
>>> edge_attr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
>>> to_dense_adj(edge_index, batch, edge_attr)
tensor([[[1., 2.],
        [3., 0.]],
        [[0., 4.],
        [5., 0.]]])
```


**to_nested_tensor(*x: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *batch: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *ptr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *batch_size: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/nested.html#to_nested_tensor)**
: Given a contiguous batch of tensors
$\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}$
(with $N_i$ indicating the number of elements in example $i$),
creates a [nested PyTorch tensor](https://pytorch.org/docs/stable/nested.html).
Reverse operation of from_nested_tensor().


**Parameters:**
: - **x** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The input tensor
$\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}$.
- **batch** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The batch vector
$\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N$, which assigns each
element to a specific example. Must be ordered.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **ptr** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – Alternative representation of
`batch` in compressed format. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **batch_size** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The batch size $B$.
(default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


**from_nested_tensor(*x: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *return_batch: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]][[source]](../_modules/torch_geometric/utils/nested.html#from_nested_tensor)**
: Given a [nested PyTorch tensor](https://pytorch.org/docs/stable/nested.html), creates a contiguous
batch of tensors
$\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}$, and
optionally a batch vector which assigns each element to a specific example.
Reverse operation of to_nested_tensor().


**Parameters:**
: - **x** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The nested input tensor. The size of nested tensors
need to match except for the first dimension.
- **return_batch** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will also return
the batch vector $\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N$.
(default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: `Union`[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]


**dense_to_sparse(*adj: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *mask: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/sparse.html#dense_to_sparse)**
: Converts a dense adjacency matrix to a sparse adjacency matrix defined
by edge indices and edge attributes.


**Parameters:**
: - **adj** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The dense adjacency matrix of shape
`[num_nodes, num_nodes]` or
`[batch_size, num_nodes, num_nodes]`.
- **mask** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – A boolean tensor of shape
`[batch_size, num_nodes]` holding information about which
nodes are in each example are valid. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: (`LongTensor`, `Tensor`)


Examples


```
>>> # For a single adjacency matrix:
>>> adj = torch.tensor([[3, 1],
...                     [2, 0]])
>>> dense_to_sparse(adj)
(tensor([[0, 0, 1],
        [0, 1, 0]]),
tensor([3, 1, 2]))
```


```
>>> # For two adjacency matrixes:
>>> adj = torch.tensor([[[3, 1],
...                      [2, 0]],
...                     [[0, 1],
...                      [0, 2]]])
>>> dense_to_sparse(adj)
(tensor([[0, 0, 1, 2, 3],
        [0, 1, 0, 3, 3]]),
tensor([3, 1, 2, 1, 2]))
```


```
>>> # First graph with two nodes, second with three:
>>> adj = torch.tensor([[
...         [3, 1, 0],
...         [2, 0, 0],
...         [0, 0, 0]
...     ], [
...         [0, 1, 0],
...         [0, 2, 3],
...         [0, 5, 0]
...     ]])
>>> mask = torch.tensor([
...         [True, True, False],
...         [True, True, True]
...     ])
>>> dense_to_sparse(adj, mask)
(tensor([[0, 0, 1, 2, 3, 3, 4],
        [0, 1, 0, 3, 3, 4, 3]]),
tensor([3, 1, 2, 1, 2, 3, 5]))
```


**is_torch_sparse_tensor(*src: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [bool](https://docs.python.org/3/library/functions.html#bool)[[source]](../_modules/torch_geometric/utils/sparse.html#is_torch_sparse_tensor)**
: Returns [True](https://docs.python.org/3/library/constants.html#True) if the input `src` is a
`torch.sparse.Tensor` (in any sparse layout).


**Parameters:**
: **src** (*Any*) – The input object to be checked.

**Return type:**
: [bool](https://docs.python.org/3/library/functions.html#bool)


**is_sparse(*src: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [bool](https://docs.python.org/3/library/functions.html#bool)[[source]](../_modules/torch_geometric/utils/sparse.html#is_sparse)**
: Returns [True](https://docs.python.org/3/library/constants.html#True) if the input `src` is of type
`torch.sparse.Tensor` (in any sparse layout) or of type
`torch_sparse.SparseTensor`.


**Parameters:**
: **src** (*Any*) – The input object to be checked.

**Return type:**
: [bool](https://docs.python.org/3/library/functions.html#bool)


**to_torch_coo_tensor(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *size: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)], [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)]]]] = None*, *is_coalesced: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/sparse.html#to_torch_coo_tensor)**
: Converts a sparse adjacency matrix defined by edge indices and edge
attributes to a `torch.sparse.Tensor` with layout
torch.sparse_coo.
See to_edge_index() for the reverse operation.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **edge_attr** (*Tensor**, **optional*) – The edge attributes.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **size** ([int](https://docs.python.org/3/library/functions.html#int)* or **(*[int](https://docs.python.org/3/library/functions.html#int)*, *[int](https://docs.python.org/3/library/functions.html#int)*)**, **optional*) – The size of the sparse matrix.
If given as an integer, will create a quadratic sparse matrix.
If set to [None](https://docs.python.org/3/library/constants.html#None), will infer a quadratic sparse matrix based
on `edge_index.max() + 1`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **is_coalesced** ([bool](https://docs.python.org/3/library/functions.html#bool)) – If set to [True](https://docs.python.org/3/library/constants.html#True), will assume that
`edge_index` is already coalesced and thus avoids expensive
computation. (default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: `torch.sparse.Tensor`


Example


```
>>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
...                            [1, 0, 2, 1, 3, 2]])
>>> to_torch_coo_tensor(edge_index)
tensor(indices=tensor([[0, 1, 1, 2, 2, 3],
                       [1, 0, 2, 1, 3, 2]]),
       values=tensor([1., 1., 1., 1., 1., 1.]),
       size=(4, 4), nnz=6, layout=torch.sparse_coo)
```


**to_torch_csr_tensor(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *size: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)], [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)]]]] = None*, *is_coalesced: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/sparse.html#to_torch_csr_tensor)**
: Converts a sparse adjacency matrix defined by edge indices and edge
attributes to a `torch.sparse.Tensor` with layout
torch.sparse_csr.
See to_edge_index() for the reverse operation.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **edge_attr** (*Tensor**, **optional*) – The edge attributes.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **size** ([int](https://docs.python.org/3/library/functions.html#int)* or **(*[int](https://docs.python.org/3/library/functions.html#int)*, *[int](https://docs.python.org/3/library/functions.html#int)*)**, **optional*) – The size of the sparse matrix.
If given as an integer, will create a quadratic sparse matrix.
If set to [None](https://docs.python.org/3/library/constants.html#None), will infer a quadratic sparse matrix based
on `edge_index.max() + 1`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **is_coalesced** ([bool](https://docs.python.org/3/library/functions.html#bool)) – If set to [True](https://docs.python.org/3/library/constants.html#True), will assume that
`edge_index` is already coalesced and thus avoids expensive
computation. (default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: `torch.sparse.Tensor`


Example


```
>>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
...                            [1, 0, 2, 1, 3, 2]])
>>> to_torch_csr_tensor(edge_index)
tensor(crow_indices=tensor([0, 1, 3, 5, 6]),
       col_indices=tensor([1, 0, 2, 1, 3, 2]),
       values=tensor([1., 1., 1., 1., 1., 1.]),
       size=(4, 4), nnz=6, layout=torch.sparse_csr)
```


**to_torch_csc_tensor(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *size: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)], [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)]]]] = None*, *is_coalesced: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/sparse.html#to_torch_csc_tensor)**
: Converts a sparse adjacency matrix defined by edge indices and edge
attributes to a `torch.sparse.Tensor` with layout
torch.sparse_csc.
See to_edge_index() for the reverse operation.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **edge_attr** (*Tensor**, **optional*) – The edge attributes.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **size** ([int](https://docs.python.org/3/library/functions.html#int)* or **(*[int](https://docs.python.org/3/library/functions.html#int)*, *[int](https://docs.python.org/3/library/functions.html#int)*)**, **optional*) – The size of the sparse matrix.
If given as an integer, will create a quadratic sparse matrix.
If set to [None](https://docs.python.org/3/library/constants.html#None), will infer a quadratic sparse matrix based
on `edge_index.max() + 1`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **is_coalesced** ([bool](https://docs.python.org/3/library/functions.html#bool)) – If set to [True](https://docs.python.org/3/library/constants.html#True), will assume that
`edge_index` is already coalesced and thus avoids expensive
computation. (default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: `torch.sparse.Tensor`


Example


```
>>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
...                            [1, 0, 2, 1, 3, 2]])
>>> to_torch_csc_tensor(edge_index)
tensor(ccol_indices=tensor([0, 1, 3, 5, 6]),
       row_indices=tensor([1, 0, 2, 1, 3, 2]),
       values=tensor([1., 1., 1., 1., 1., 1.]),
       size=(4, 4), nnz=6, layout=torch.sparse_csc)
```


**to_torch_sparse_tensor(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *size: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)], [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)]]]] = None*, *is_coalesced: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *layout: [layout](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.layout) = torch.sparse_coo*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/sparse.html#to_torch_sparse_tensor)**
: Converts a sparse adjacency matrix defined by edge indices and edge
attributes to a `torch.sparse.Tensor` with custom `layout`.
See to_edge_index() for the reverse operation.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **edge_attr** (*Tensor**, **optional*) – The edge attributes.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **size** ([int](https://docs.python.org/3/library/functions.html#int)* or **(*[int](https://docs.python.org/3/library/functions.html#int)*, *[int](https://docs.python.org/3/library/functions.html#int)*)**, **optional*) – The size of the sparse matrix.
If given as an integer, will create a quadratic sparse matrix.
If set to [None](https://docs.python.org/3/library/constants.html#None), will infer a quadratic sparse matrix based
on `edge_index.max() + 1`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **is_coalesced** ([bool](https://docs.python.org/3/library/functions.html#bool)) – If set to [True](https://docs.python.org/3/library/constants.html#True), will assume that
`edge_index` is already coalesced and thus avoids expensive
computation. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **layout** ([torch.layout](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.layout)*, **optional*) – The layout of the output sparse tensor
(`torch.sparse_coo`, `torch.sparse_csr`,
`torch.sparse_csc`). (default: `torch.sparse_coo`)

**Return type:**
: `torch.sparse.Tensor`


**to_edge_index(*adj: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), SparseTensor]*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/sparse.html#to_edge_index)**
: Converts a `torch.sparse.Tensor` or a
`torch_sparse.SparseTensor` to edge indices and edge attributes.


**Parameters:**
: **adj** (*torch.sparse.Tensor** or **SparseTensor*) – The adjacency matrix.

**Return type:**
: ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor))


Example


```
>>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
...                            [1, 0, 2, 1, 3, 2]])
>>> adj = to_torch_coo_tensor(edge_index)
>>> to_edge_index(adj)
(tensor([[0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]]),
tensor([1., 1., 1., 1., 1., 1.]))
```


**spmm(*src: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), SparseTensor]*, *other: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *reduce: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'sum'*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/_spmm.html#spmm)**
: Matrix product of sparse matrix with dense matrix.


**Parameters:**
: - **src** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)* or **torch_sparse.SparseTensor** or *[EdgeIndex](../generated/torch_geometric.EdgeIndex.html#torch_geometric.EdgeIndex)) – The input sparse matrix which can be a
PyG `torch_sparse.SparseTensor`,
a PyTorch `torch.sparse.Tensor` or
a PyG `EdgeIndex`.
- **other** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The input dense matrix.
- **reduce** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The reduce operation to use
(`"sum"`, `"mean"`, `"min"`, `"max"`).
(default: `"sum"`)

**Return type:**
: `Tensor`


**unbatch(*src: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *batch: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *dim: [int](https://docs.python.org/3/library/functions.html#int) = 0*, *batch_size: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/_unbatch.html#unbatch)**
: Splits `src` according to a `batch` vector along dimension
`dim`.


**Parameters:**
: - **src** (*Tensor*) – The source tensor.
- **batch** (*LongTensor*) – The batch vector
$\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N$, which assigns each
entry in `src` to a specific example. Must be ordered.
- **dim** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The dimension along which to split the `src`
tensor. (default: `0`)
- **batch_size** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The batch size. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: `List[Tensor]`


Example


```
>>> src = torch.arange(7)
>>> batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
>>> unbatch(src, batch)
(tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))
```


**unbatch_edge_index(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *batch: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *batch_size: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/_unbatch.html#unbatch_edge_index)**
: Splits the `edge_index` according to a `batch` vector.


**Parameters:**
: - **edge_index** (*Tensor*) – The edge_index tensor. Must be ordered.
- **batch** (*LongTensor*) – The batch vector
$\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N$, which assigns each
node to a specific example. Must be ordered.
- **batch_size** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The batch size. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: `List[Tensor]`


Example


```
>>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
...                            [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]])
>>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
>>> unbatch_edge_index(edge_index, batch)
(tensor([[0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]]),
tensor([[0, 1, 1, 2],
        [1, 0, 2, 1]]))
```


**one_hot(*index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *num_classes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *dtype: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[dtype](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.dtype)] = None*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/_one_hot.html#one_hot)**
: Taskes a one-dimensional `index` tensor and returns a one-hot
encoded representation of it with shape `[*, num_classes]` that has
zeros everywhere except where the index of last dimension matches the
corresponding value of the input tensor, in which case it will be `1`.


> **Note:** This is a more memory-efficient version of
`torch.nn.functional.one_hot()` as you can customize the output
`dtype`.


**Parameters:**
: - **index** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The one-dimensional input tensor.
- **num_classes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The total number of classes. If set to
[None](https://docs.python.org/3/library/constants.html#None), the number of classes will be inferred as one greater
than the largest class value in the input tensor.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **dtype** ([torch.dtype](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.dtype)*, **optional*) – The `dtype` of the output tensor.

**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


**normalized_cut(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/_normalized_cut.html#normalized_cut)**
: Computes the normalized cut $\mathbf{e}_{i,j} \cdot
\left( \frac{1}{\deg(i)} + \frac{1}{\deg(j)} \right)$ of a weighted graph
given by edge indices and edge attributes.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **edge_attr** (*Tensor*) – Edge weights or multi-dimensional edge features.
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max_val + 1` of `edge_index`. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: `Tensor`


Example


```
>>> edge_index = torch.tensor([[1, 1, 2, 3],
...                            [3, 3, 1, 2]])
>>> edge_attr = torch.tensor([1., 1., 1., 1.])
>>> normalized_cut(edge_index, edge_attr)
tensor([1.5000, 1.5000, 2.0000, 1.5000])
```


**grid(*height: [int](https://docs.python.org/3/library/functions.html#int)*, *width: [int](https://docs.python.org/3/library/functions.html#int)*, *dtype: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[dtype](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.dtype)] = None*, *device: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[device](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.device)] = None*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/_grid.html#grid)**
: Returns the edge indices of a two-dimensional grid graph with height
`height` and width `width` and its node positions.


**Parameters:**
: - **height** ([int](https://docs.python.org/3/library/functions.html#int)) – The height of the grid.
- **width** ([int](https://docs.python.org/3/library/functions.html#int)) – The width of the grid.
- **dtype** ([torch.dtype](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.dtype)*, **optional*) – The desired data type of the returned
position tensor. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **device** ([torch.device](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.device)*, **optional*) – The desired device of the returned
tensors. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: (`LongTensor`, `Tensor`)


Example


```
>>> (row, col), pos = grid(height=2, width=2)
>>> row
tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
>>> col
tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
>>> pos
tensor([[0., 1.],
        [1., 1.],
        [0., 0.],
        [1., 0.]])
```


**geodesic_distance(*pos: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *face: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *src: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *dst: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *norm: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, *max_distance: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[float](https://docs.python.org/3/library/functions.html#float)] = None*, *num_workers: [int](https://docs.python.org/3/library/functions.html#int) = 0*, ***kwargs: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/geodesic.html#geodesic_distance)**
: Computes (normalized) geodesic distances of a mesh given by `pos`
and `face`. If `src` and `dst` are given, this method only
computes the geodesic distances for the respective source and target
node-pairs.


> **Note:** This function requires the `gdist` package.
To install, run `pip install cython && pip install gdist`.


**Parameters:**
: - **pos** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The node positions.
- **face** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The face indices.
- **src** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – If given, only compute geodesic distances
for the specified source indices. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **dst** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – If given, only compute geodesic distances
for the specified target indices. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **norm** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – Normalizes geodesic distances by
$\sqrt{\textrm{area}(\mathcal{M})}$. (default: [True](https://docs.python.org/3/library/constants.html#True))
- **max_distance** ([float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – If given, only yields results for
geodesic distances less than `max_distance`. This will speed
up runtime dramatically. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_workers** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – How many subprocesses to use for
calculating geodesic distances.
`0` means that computation takes place in the main process.
`-1` means that the available amount of CPU cores is used.
(default: `0`)

**Return type:**
: `Tensor`


Example


```
>>> pos = torch.tensor([[0.0, 0.0, 0.0],
...                     [2.0, 0.0, 0.0],
...                     [0.0, 2.0, 0.0],
...                     [2.0, 2.0, 0.0]])
>>> face = torch.tensor([[0, 0],
...                      [1, 2],
...                      [3, 3]])
>>> geodesic_distance(pos, face)
[[0, 1, 1, 1.4142135623730951],
[1, 0, 1.4142135623730951, 1],
[1, 1.4142135623730951, 0, 1],
[1.4142135623730951, 1, 1, 0]]
```


**to_scipy_sparse_matrix(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)[[source]](../_modules/torch_geometric/utils/convert.html#to_scipy_sparse_matrix)**
: Converts a graph given by edge indices and edge attributes to a scipy
sparse matrix.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **edge_attr** (*Tensor**, **optional*) – Edge weights or multi-dimensional
edge features. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max_val + 1` of `index`. (default: [None](https://docs.python.org/3/library/constants.html#None))


Examples


```
>>> edge_index = torch.tensor([
...     [0, 1, 1, 2, 2, 3],
...     [1, 0, 2, 1, 3, 2],
... ])
>>> to_scipy_sparse_matrix(edge_index)
<4x4 sparse matrix of type '<class 'numpy.float32'>'
    with 6 stored elements in COOrdinate format>
```


**Return type:**
: [Any](https://docs.python.org/3/library/typing.html#typing.Any)


**from_scipy_sparse_matrix(*A: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/convert.html#from_scipy_sparse_matrix)**
: Converts a scipy sparse matrix to edge indices and edge attributes.


**Parameters:**
: **A** (*scipy.sparse*) – A sparse matrix.


Examples


```
>>> edge_index = torch.tensor([
...     [0, 1, 1, 2, 2, 3],
...     [1, 0, 2, 1, 3, 2],
... ])
>>> adj = to_scipy_sparse_matrix(edge_index)
>>> # `edge_index` and `edge_weight` are both returned
>>> from_scipy_sparse_matrix(adj)
(tensor([[0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]]),
tensor([1., 1., 1., 1., 1., 1.]))
```


**Return type:**
: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]


**to_networkx(*data: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)]*, *node_attrs: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*, *edge_attrs: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*, *graph_attrs: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*, *to_undirected: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[bool](https://docs.python.org/3/library/functions.html#bool), [str](https://docs.python.org/3/library/stdtypes.html#str)]] = False*, *to_multi: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *remove_self_loops: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)[[source]](../_modules/torch_geometric/utils/convert.html#to_networkx)**
: Converts a [torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) instance to a
`networkx.Graph` if to_undirected is set to [True](https://docs.python.org/3/library/constants.html#True), or
a directed `networkx.DiGraph` otherwise.


**Parameters:**
: - **data** ([torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)* or *[torch_geometric.data.HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)) – A
homogeneous or heterogeneous data object.
- **node_attrs** (*iterable of str**, **optional*) – The node attributes to be
copied. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **edge_attrs** (*iterable of str**, **optional*) – The edge attributes to be
copied. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **graph_attrs** (*iterable of str**, **optional*) – The graph attributes to be
copied. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **to_undirected** ([bool](https://docs.python.org/3/library/functions.html#bool)* or *[str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will
return a `networkx.Graph` instead of a
`networkx.DiGraph`.
By default, will include all edges and make them undirected.
If set to `"upper"`, the undirected graph will only correspond
to the upper triangle of the input adjacency matrix.
If set to `"lower"`, the undirected graph will only correspond
to the lower triangle of the input adjacency matrix.
Only applicable in case the `data` object holds a homogeneous
graph. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **to_multi** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – if set to [True](https://docs.python.org/3/library/constants.html#True), will return a
`networkx.MultiGraph` or a `networkx:MultiDiGraph`
(depending on the to_undirected option), which will not drop
duplicated edges that may exist in `data`.
(default: [False](https://docs.python.org/3/library/constants.html#False))
- **remove_self_loops** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will not
include self-loops in the resulting graph. (default: [False](https://docs.python.org/3/library/constants.html#False))


Examples


```
>>> edge_index = torch.tensor([
...     [0, 1, 1, 2, 2, 3],
...     [1, 0, 2, 1, 3, 2],
... ])
>>> data = Data(edge_index=edge_index, num_nodes=4)
>>> to_networkx(data)
<networkx.classes.digraph.DiGraph at 0x2713fdb40d0>
```


**Return type:**
: [Any](https://docs.python.org/3/library/typing.html#typing.Any)


**from_networkx(*G: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*, *group_node_attrs: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)], [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['all']]] = None*, *group_edge_attrs: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)], [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['all']]] = None*) → [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)[[source]](../_modules/torch_geometric/utils/convert.html#from_networkx)**
: Converts a `networkx.Graph` or `networkx.DiGraph` to a
[torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) instance.


**Parameters:**
: - **G** (*networkx.Graph** or **networkx.DiGraph*) – A networkx graph.
- **group_node_attrs** (*List**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*] or **"all"**, **optional*) – The node attributes to
be concatenated and added to `data.x`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **group_edge_attrs** (*List**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*] or **"all"**, **optional*) – The edge attributes to
be concatenated and added to `data.edge_attr`.
(default: [None](https://docs.python.org/3/library/constants.html#None))


> **Note:** All `group_node_attrs` and `group_edge_attrs` values must
be numeric.


Examples


```
>>> edge_index = torch.tensor([
...     [0, 1, 1, 2, 2, 3],
...     [1, 0, 2, 1, 3, 2],
... ])
>>> data = Data(edge_index=edge_index, num_nodes=4)
>>> g = to_networkx(data)
>>> # A `Data` object is returned
>>> from_networkx(g)
Data(edge_index=[2, 6], num_nodes=4)
```


**Return type:**
: [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)


**to_networkit(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_weight: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *directed: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)[[source]](../_modules/torch_geometric/utils/convert.html#to_networkit)**
: Converts a `(edge_index, edge_weight)` tuple to a
`networkit.Graph`.


**Parameters:**
: - **edge_index** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The edge indices of the graph.
- **edge_weight** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The edge weights of the graph.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes in the graph.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **directed** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [False](https://docs.python.org/3/library/constants.html#False), the graph will be
undirected. (default: [True](https://docs.python.org/3/library/constants.html#True))

**Return type:**
: [Any](https://docs.python.org/3/library/typing.html#typing.Any)


**from_networkit(*g: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]][[source]](../_modules/torch_geometric/utils/convert.html#from_networkit)**
: Converts a `networkit.Graph` to a
`(edge_index, edge_weight)` tuple.
If the `networkit.Graph` is not weighted, the returned
`edge_weight` will be [None](https://docs.python.org/3/library/constants.html#None).


**Parameters:**
: **g** (*networkkit.graph.Graph*) – A `networkit` graph object.

**Return type:**
: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]


**to_trimesh(*data: [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)[[source]](../_modules/torch_geometric/utils/convert.html#to_trimesh)**
: Converts a [torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) instance to a
`trimesh.Trimesh`.


**Parameters:**
: **data** ([torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)) – The data object.


Example


```
>>> pos = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
...                    dtype=torch.float)
>>> face = torch.tensor([[0, 1, 2], [1, 2, 3]]).t()
```


```
>>> data = Data(pos=pos, face=face)
>>> to_trimesh(data)
<trimesh.Trimesh(vertices.shape=(4, 3), faces.shape=(2, 3))>
```


**Return type:**
: [Any](https://docs.python.org/3/library/typing.html#typing.Any)


**from_trimesh(*mesh: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)[[source]](../_modules/torch_geometric/utils/convert.html#from_trimesh)**
: Converts a `trimesh.Trimesh` to a
[torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) instance.


**Parameters:**
: **mesh** (*trimesh.Trimesh*) – A `trimesh` mesh.


Example


```
>>> pos = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
...                    dtype=torch.float)
>>> face = torch.tensor([[0, 1, 2], [1, 2, 3]]).t()
```


```
>>> data = Data(pos=pos, face=face)
>>> mesh = to_trimesh(data)
>>> from_trimesh(mesh)
Data(pos=[4, 3], face=[3, 2])
```


**Return type:**
: [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)


**to_cugraph(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_weight: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *relabel_nodes: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, *directed: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)[[source]](../_modules/torch_geometric/utils/convert.html#to_cugraph)**
: Converts a graph given by `edge_index` and optional
`edge_weight` into a `cugraph` graph object.


**Parameters:**
: - **edge_index** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The edge indices of the graph.
- **edge_weight** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The edge weights of the graph.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **relabel_nodes** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True),
`cugraph` will remove any isolated nodes, leading to a
relabeling of nodes. (default: [True](https://docs.python.org/3/library/constants.html#True))
- **directed** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [False](https://docs.python.org/3/library/constants.html#False), the graph will be
undirected. (default: [True](https://docs.python.org/3/library/constants.html#True))

**Return type:**
: [Any](https://docs.python.org/3/library/typing.html#typing.Any)


**from_cugraph(*g: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]][[source]](../_modules/torch_geometric/utils/convert.html#from_cugraph)**
: Converts a `cugraph` graph object into `edge_index` and
optional `edge_weight` tensors.


**Parameters:**
: **g** (*cugraph.Graph*) – A `cugraph` graph object.

**Return type:**
: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]


**to_dgl(*data: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)]*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)[[source]](../_modules/torch_geometric/utils/convert.html#to_dgl)**
: Converts a [torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) or
[torch_geometric.data.HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData) instance to a `dgl` graph
object.


**Parameters:**
: **data** ([torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)* or *[torch_geometric.data.HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)) – The data object.


Example


```
>>> edge_index = torch.tensor([[0, 1, 1, 2, 3, 0], [1, 0, 2, 1, 4, 4]])
>>> x = torch.randn(5, 3)
>>> edge_attr = torch.randn(6, 2)
>>> data = Data(x=x, edge_index=edge_index, edge_attr=y)
>>> g = to_dgl(data)
>>> g
Graph(num_nodes=5, num_edges=6,
    ndata_schemes={'x': Scheme(shape=(3,))}
    edata_schemes={'edge_attr': Scheme(shape=(2, ))})
```


```
>>> data = HeteroData()
>>> data['paper'].x = torch.randn(5, 3)
>>> data['author'].x = torch.ones(5, 3)
>>> edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
>>> data['author', 'cites', 'paper'].edge_index = edge_index
>>> g = to_dgl(data)
>>> g
Graph(num_nodes={'author': 5, 'paper': 5},
    num_edges={('author', 'cites', 'paper'): 5},
    metagraph=[('author', 'paper', 'cites')])
```


**Return type:**
: [Any](https://docs.python.org/3/library/typing.html#typing.Any)


**from_dgl(*g: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)][[source]](../_modules/torch_geometric/utils/convert.html#from_dgl)**
: Converts a `dgl` graph object to a
[torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) or
[torch_geometric.data.HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData) instance.


**Parameters:**
: **g** (*dgl.DGLGraph*) – The `dgl` graph object.


Example


```
>>> g = dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0]))
>>> g.ndata['x'] = torch.randn(g.num_nodes(), 3)
>>> g.edata['edge_attr'] = torch.randn(g.num_edges(), 2)
>>> data = from_dgl(g)
>>> data
Data(x=[6, 3], edge_attr=[4, 2], edge_index=[2, 4])
```


```
>>> g = dgl.heterograph({
>>> g = dgl.heterograph({
...     ('author', 'writes', 'paper'): ([0, 1, 1, 2, 3, 3, 4],
...                                     [0, 0, 1, 1, 1, 2, 2])})
>>> g.nodes['author'].data['x'] = torch.randn(5, 3)
>>> g.nodes['paper'].data['x'] = torch.randn(5, 3)
>>> data = from_dgl(g)
>>> data
HeteroData(
author={ x=[5, 3] },
paper={ x=[3, 3] },
(author, writes, paper)={ edge_index=[2, 7] }
)
```


**Return type:**
: `Union`[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)]


**from_rdmol(*mol: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)[[source]](../_modules/torch_geometric/utils/smiles.html#from_rdmol)**
: Converts a `rdkit.Chem.Mol` instance to a
[torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) instance.


**Parameters:**
: **mol** (*rdkit.Chem.Mol*) – The `rdkit` molecule.

**Return type:**
: [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)


**to_rdmol(*data: [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)*, *kekulize: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)[[source]](../_modules/torch_geometric/utils/smiles.html#to_rdmol)**
: Converts a [torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) instance to a
`rdkit.Chem.Mol` instance.


**Parameters:**
: - **data** ([torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)) – The molecular graph data.
- **kekulize** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), converts aromatic
bonds to single/double bonds. (default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: [Any](https://docs.python.org/3/library/typing.html#typing.Any)


**from_smiles(*smiles: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *with_hydrogen: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *kekulize: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)[[source]](../_modules/torch_geometric/utils/smiles.html#from_smiles)**
: Converts a SMILES string to a [torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)
instance.


**Parameters:**
: - **smiles** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – The SMILES string.
- **with_hydrogen** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will store
hydrogens in the molecule graph. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **kekulize** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), converts aromatic
bonds to single/double bonds. (default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)


**to_smiles(*data: [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)*, *kekulize: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [str](https://docs.python.org/3/library/stdtypes.html#str)[[source]](../_modules/torch_geometric/utils/smiles.html#to_smiles)**
: Converts a [torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) instance to a SMILES
string.


**Parameters:**
: - **data** ([torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)) – The molecular graph.
- **kekulize** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), converts aromatic
bonds to single/double bonds. (default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: [str](https://docs.python.org/3/library/stdtypes.html#str)


**erdos_renyi_graph(*num_nodes: [int](https://docs.python.org/3/library/functions.html#int)*, *edge_prob: [float](https://docs.python.org/3/library/functions.html#float)*, *directed: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/random.html#erdos_renyi_graph)**
: Returns the `edge_index` of a random Erdos-Renyi graph.


**Parameters:**
: - **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)) – The number of nodes.
- **edge_prob** ([float](https://docs.python.org/3/library/functions.html#float)) – Probability of an edge.
- **directed** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will return a
directed graph. (default: [False](https://docs.python.org/3/library/constants.html#False))


Examples


```
>>> erdos_renyi_graph(5, 0.2, directed=False)
tensor([[0, 1, 1, 4],
        [1, 0, 4, 1]])
```


```
>>> erdos_renyi_graph(5, 0.2, directed=True)
tensor([[0, 1, 3, 3, 4, 4],
        [4, 3, 1, 2, 1, 3]])
```


**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


**stochastic_blockmodel_graph(*block_sizes: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *edge_probs: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[float](https://docs.python.org/3/library/functions.html#float)]], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *directed: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/random.html#stochastic_blockmodel_graph)**
: Returns the `edge_index` of a stochastic blockmodel graph.


**Parameters:**
: - **block_sizes** (*[*[int](https://docs.python.org/3/library/functions.html#int)*] or **LongTensor*) – The sizes of blocks.
- **edge_probs** (*[**[*[float](https://docs.python.org/3/library/functions.html#float)*]**] or **FloatTensor*) – The density of edges going
from each block to each other block. Must be symmetric if the
graph is undirected.
- **directed** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will return a
directed graph. (default: [False](https://docs.python.org/3/library/constants.html#False))


Examples


```
>>> block_sizes = [2, 2, 4]
>>> edge_probs = [[0.25, 0.05, 0.02],
...               [0.05, 0.35, 0.07],
...               [0.02, 0.07, 0.40]]
>>> stochastic_blockmodel_graph(block_sizes, edge_probs,
...                             directed=False)
tensor([[2, 4, 4, 5, 5, 6, 7, 7],
        [5, 6, 7, 2, 7, 4, 4, 5]])
```


```
>>> stochastic_blockmodel_graph(block_sizes, edge_probs,
...                             directed=True)
tensor([[0, 2, 3, 4, 4, 5, 5],
        [3, 4, 1, 5, 6, 6, 7]])
```


**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


**barabasi_albert_graph(*num_nodes: [int](https://docs.python.org/3/library/functions.html#int)*, *num_edges: [int](https://docs.python.org/3/library/functions.html#int)*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/random.html#barabasi_albert_graph)**
: Returns the `edge_index` of a Barabasi-Albert preferential
attachment model, where a graph of `num_nodes` nodes grows by
attaching new nodes with `num_edges` edges that are preferentially
attached to existing nodes with high degree.


**Parameters:**
: - **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)) – The number of nodes.
- **num_edges** ([int](https://docs.python.org/3/library/functions.html#int)) – The number of edges from a new node to existing nodes.


Example


```
>>> barabasi_albert_graph(num_nodes=4, num_edges=3)
tensor([[0, 0, 0, 1, 1, 2, 2, 3],
        [1, 2, 3, 0, 2, 0, 1, 0]])
```


**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


**negative_sampling(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)]]] = None*, *num_neg_samples: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)]] = None*, *method: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'sparse'*, *force_undirected: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/_negative_sampling.html#negative_sampling)**
: Samples random negative edges of a graph given by `edge_index`.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)* or **Tuple**[*[int](https://docs.python.org/3/library/functions.html#int)*, *[int](https://docs.python.org/3/library/functions.html#int)*]**, **optional*) – The number of nodes,
*i.e.* `max_val + 1` of `edge_index`.
If given as a tuple, then `edge_index` is interpreted as a
bipartite graph with shape `(num_src_nodes, num_dst_nodes)`.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_neg_samples** ([int](https://docs.python.org/3/library/functions.html#int)* or *[float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – The (approximate) number of
negative samples to return. If set to a floating-point value, it
represents the ratio of negative samples to generate based on the
number of positive edges. If set to [None](https://docs.python.org/3/library/constants.html#None), will try to
return a negative edge for every positive edge.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **method** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The method to use for negative sampling,
*i.e.* `"sparse"` or `"dense"`.
This is a memory/runtime trade-off.
`"sparse"` will work on any graph of any size, while
`"dense"` can perform faster true-negative checks.
(default: `"sparse"`)
- **force_undirected** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), sampled
negative edges will be undirected. (default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: LongTensor


Examples


```
>>> # Standard usage
>>> edge_index = torch.as_tensor([[0, 0, 1, 2],
...                               [0, 1, 2, 3]])
>>> negative_sampling(edge_index)
tensor([[3, 0, 0, 3],
        [2, 3, 2, 1]])
```


```
>>> negative_sampling(edge_index, num_nodes=(3, 4),
...                   num_neg_samples=0.5)  # 50% of positive edges
tensor([[0, 3],
        [3, 0]])
```


```
>>> # For bipartite graph
>>> negative_sampling(edge_index, num_nodes=(3, 4))
tensor([[0, 2, 2, 1],
        [2, 2, 1, 3]])
```


**batched_negative_sampling(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *batch: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]*, *num_neg_samples: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)]] = None*, *method: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'sparse'*, *force_undirected: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/utils/_negative_sampling.html#batched_negative_sampling)**
: Samples random negative edges of multiple graphs given by
`edge_index` and `batch`.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **batch** (*LongTensor** or **Tuple**[**LongTensor**, **LongTensor**]*) – Batch vector
$\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N$, which assigns each
node to a specific example.
If given as a tuple, then `edge_index` is interpreted as a
bipartite graph connecting two different node types.
- **num_neg_samples** ([int](https://docs.python.org/3/library/functions.html#int)* or *[float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – The number of negative
samples to return. If set to [None](https://docs.python.org/3/library/constants.html#None), will try to return a
negative edge for every positive edge. If float, it will generate
`num_neg_samples * num_edges` negative samples.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **method** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The method to use for negative sampling,
*i.e.* `"sparse"` or `"dense"`.
This is a memory/runtime trade-off.
`"sparse"` will work on any graph of any size, while
`"dense"` can perform faster true-negative checks.
(default: `"sparse"`)
- **force_undirected** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), sampled
negative edges will be undirected. (default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: LongTensor


Examples


```
>>> # Standard usage
>>> edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])
>>> edge_index = torch.cat([edge_index, edge_index + 4], dim=1)
>>> edge_index
tensor([[0, 0, 1, 2, 4, 4, 5, 6],
        [0, 1, 2, 3, 4, 5, 6, 7]])
>>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
>>> batched_negative_sampling(edge_index, batch)
tensor([[3, 1, 3, 2, 7, 7, 6, 5],
        [2, 0, 1, 1, 5, 6, 4, 4]])
```


```
>>> # Using float multiplier for negative samples
>>> batched_negative_sampling(edge_index, batch, num_neg_samples=1.5)
tensor([[3, 1, 3, 2, 7, 7, 6, 5, 2, 0, 1, 1],
        [2, 0, 1, 1, 5, 6, 4, 4, 3, 2, 3, 0]])
```


```
>>> # For bipartite graph
>>> edge_index1 = torch.as_tensor([[0, 0, 1, 1], [0, 1, 2, 3]])
>>> edge_index2 = edge_index1 + torch.tensor([[2], [4]])
>>> edge_index3 = edge_index2 + torch.tensor([[2], [4]])
>>> edge_index = torch.cat([edge_index1, edge_index2,
...                         edge_index3], dim=1)
>>> edge_index
tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
        [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]])
>>> src_batch = torch.tensor([0, 0, 1, 1, 2, 2])
>>> dst_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
>>> batched_negative_sampling(edge_index,
...                           (src_batch, dst_batch))
tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
        [ 2,  3,  0,  1,  6,  7,  4,  5, 10, 11,  8,  9]])
```


**structured_negative_sampling(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *contains_neg_self_loops: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/_negative_sampling.html#structured_negative_sampling)**
: Samples a negative edge `(i,k)` for every positive edge
`(i,j)` in the graph given by `edge_index`, and returns it as a
tuple of the form `(i,j,k)`.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max_val + 1` of `edge_index`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **contains_neg_self_loops** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to
[False](https://docs.python.org/3/library/constants.html#False), sampled negative edges will not contain self loops.
(default: [True](https://docs.python.org/3/library/constants.html#True))

**Return type:**
: (LongTensor, LongTensor, LongTensor)


Example


```
>>> edge_index = torch.as_tensor([[0, 0, 1, 2],
...                               [0, 1, 2, 3]])
>>> structured_negative_sampling(edge_index)
(tensor([0, 0, 1, 2]), tensor([0, 1, 2, 3]), tensor([2, 3, 0, 2]))
```


**structured_negative_sampling_feasible(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *contains_neg_self_loops: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → [bool](https://docs.python.org/3/library/functions.html#bool)[[source]](../_modules/torch_geometric/utils/_negative_sampling.html#structured_negative_sampling_feasible)**
: Returns [True](https://docs.python.org/3/library/constants.html#True) if
structured_negative_sampling() is feasible
on the graph given by `edge_index`.
structured_negative_sampling() is infeasible
if at least one node is connected to all other nodes.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes, *i.e.*
`max_val + 1` of `edge_index`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **contains_neg_self_loops** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to
[False](https://docs.python.org/3/library/constants.html#False), sampled negative edges will not contain self loops.
(default: [True](https://docs.python.org/3/library/constants.html#True))

**Return type:**
: [bool](https://docs.python.org/3/library/functions.html#bool)


Examples


```
>>> edge_index = torch.LongTensor([[0, 0, 1, 1, 2, 2, 2],
...                                [1, 2, 0, 2, 0, 1, 1]])
>>> structured_negative_sampling_feasible(edge_index, 3, False)
False
```


```
>>> structured_negative_sampling_feasible(edge_index, 3, True)
True
```


**shuffle_node(*x: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *batch: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *training: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/augmentation.html#shuffle_node)**
: Randomly shuffle the feature matrix `x` along the
first dimension.


The method returns (1) the shuffled `x`, (2) the permutation
indicating the orders of original nodes after shuffling.


**Parameters:**
: - **x** (*FloatTensor*) – The feature matrix.
- **batch** (*LongTensor**, **optional*) – Batch vector
$\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N$, which assigns each
node to a specific example. Must be ordered. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **training** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [False](https://docs.python.org/3/library/constants.html#False), this operation is a
no-op. (default: [True](https://docs.python.org/3/library/constants.html#True))

**Return type:**
: (`FloatTensor`, `LongTensor`)


Example


```
>>> # Standard case
>>> x = torch.tensor([[0, 1, 2],
...                   [3, 4, 5],
...                   [6, 7, 8],
...                   [9, 10, 11]], dtype=torch.float)
>>> x, node_perm = shuffle_node(x)
>>> x
tensor([[ 3.,  4.,  5.],
        [ 9., 10., 11.],
        [ 0.,  1.,  2.],
        [ 6.,  7.,  8.]])
>>> node_perm
tensor([1, 3, 0, 2])
```


```
>>> # For batched graphs as inputs
>>> batch = torch.tensor([0, 0, 1, 1])
>>> x, node_perm = shuffle_node(x, batch)
>>> x
tensor([[ 3.,  4.,  5.],
        [ 0.,  1.,  2.],
        [ 9., 10., 11.],
        [ 6.,  7.,  8.]])
>>> node_perm
tensor([1, 0, 3, 2])
```


**mask_feature(*x: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *p: [float](https://docs.python.org/3/library/functions.html#float) = 0.5*, *mode: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'col'*, *fill_value: [float](https://docs.python.org/3/library/functions.html#float) = 0.0*, *training: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/augmentation.html#mask_feature)**
: Randomly masks feature from the feature matrix
`x` with probability `p` using samples from
a Bernoulli distribution.


The method returns (1) the retained `x`, (2) the feature
mask broadcastable with `x` (`mode='row'` and `mode='col'`)
or with the same shape as `x` (`mode='all'`),
indicating where features are retained.


**Parameters:**
: - **x** (*FloatTensor*) – The feature matrix.
- **p** ([float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – The masking ratio. (default: `0.5`)
- **mode** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The masked scheme to use for feature masking.
(`"row"`, `"col"` or `"all"`).
If `mode='col'`, will mask entire features of all nodes
from the feature matrix. If `mode='row'`, will mask entire
nodes from the feature matrix. If `mode='all'`, will mask
individual features across all nodes. (default: `'col'`)
- **fill_value** ([float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – The value for masked features in the
output tensor. (default: `0`)
- **training** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [False](https://docs.python.org/3/library/constants.html#False), this operation is a
no-op. (default: [True](https://docs.python.org/3/library/constants.html#True))

**Return type:**
: (`FloatTensor`, `BoolTensor`)


Examples


```
>>> # Masked features are column-wise sampled
>>> x = torch.tensor([[1, 2, 3],
...                   [4, 5, 6],
...                   [7, 8, 9]], dtype=torch.float)
>>> x, feat_mask = mask_feature(x)
>>> x
tensor([[1., 0., 3.],
        [4., 0., 6.],
        [7., 0., 9.]]),
>>> feat_mask
tensor([[True, False, True]])
```


```
>>> # Masked features are row-wise sampled
>>> x, feat_mask = mask_feature(x, mode='row')
>>> x
tensor([[1., 2., 3.],
        [0., 0., 0.],
        [7., 8., 9.]]),
>>> feat_mask
tensor([[True], [False], [True]])
```


```
>>> # Masked features are uniformly sampled
>>> x, feat_mask = mask_feature(x, mode='all')
>>> x
tensor([[0., 0., 0.],
        [4., 0., 6.],
        [0., 0., 9.]])
>>> feat_mask
tensor([[False, False, False],
        [True, False,  True],
        [False, False,  True]])
```


**add_random_edge(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *p: [float](https://docs.python.org/3/library/functions.html#float) = 0.5*, *force_undirected: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[int](https://docs.python.org/3/library/functions.html#int), [int](https://docs.python.org/3/library/functions.html#int)]]] = None*, *training: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/augmentation.html#add_random_edge)**
: Randomly adds edges to `edge_index`.


The method returns (1) the retained `edge_index`, (2) the added
edge indices.


**Parameters:**
: - **edge_index** (*LongTensor*) – The edge indices.
- **p** ([float](https://docs.python.org/3/library/functions.html#float)) – Ratio of added edges to the existing edges.
(default: `0.5`)
- **force_undirected** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True),
added edges will be undirected.
(default: [False](https://docs.python.org/3/library/constants.html#False))
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **Tuple**[*[int](https://docs.python.org/3/library/functions.html#int)*]**, **optional*) – The overall number of nodes,
*i.e.* `max_val + 1`, or the number of source and
destination nodes, *i.e.* `(max_src_val + 1, max_dst_val + 1)`
of `edge_index`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **training** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [False](https://docs.python.org/3/library/constants.html#False), this operation is a
no-op. (default: [True](https://docs.python.org/3/library/constants.html#True))

**Return type:**
: (`LongTensor`, `LongTensor`)


Examples


```
>>> # Standard case
>>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
...                            [1, 0, 2, 1, 3, 2]])
>>> edge_index, added_edges = add_random_edge(edge_index, p=0.5)
>>> edge_index
tensor([[0, 1, 1, 2, 2, 3, 2, 1, 3],
        [1, 0, 2, 1, 3, 2, 0, 2, 1]])
>>> added_edges
tensor([[2, 1, 3],
        [0, 2, 1]])
```


```
>>> # The returned graph is kept undirected
>>> edge_index, added_edges = add_random_edge(edge_index, p=0.5,
...                                           force_undirected=True)
>>> edge_index
tensor([[0, 1, 1, 2, 2, 3, 2, 1, 3, 0, 2, 1],
        [1, 0, 2, 1, 3, 2, 0, 2, 1, 2, 1, 3]])
>>> added_edges
tensor([[2, 1, 3, 0, 2, 1],
        [0, 2, 1, 2, 1, 3]])
```


```
>>> # For bipartite graphs
>>> edge_index = torch.tensor([[0, 1, 2, 3, 4, 5],
...                            [2, 3, 1, 4, 2, 1]])
>>> edge_index, added_edges = add_random_edge(edge_index, p=0.5,
...                                           num_nodes=(6, 5))
>>> edge_index
tensor([[0, 1, 2, 3, 4, 5, 3, 4, 1],
        [2, 3, 1, 4, 2, 1, 1, 3, 2]])
>>> added_edges
tensor([[3, 4, 1],
        [1, 3, 2]])
```


**tree_decomposition(*mol: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*, *return_vocab: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [int](https://docs.python.org/3/library/functions.html#int)], [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [int](https://docs.python.org/3/library/functions.html#int), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]][[source]](../_modules/torch_geometric/utils/_tree_decomposition.html#tree_decomposition)**
: The tree decomposition algorithm of molecules from the
[“Junction Tree Variational Autoencoder for Molecular Graph Generation”](https://arxiv.org/abs/1802.04364) paper.
Returns the graph connectivity of the junction tree, the assignment
mapping of each atom to the clique in the junction tree, and the number
of cliques.


**Parameters:**
: - **mol** (*rdkit.Chem.Mol*) – An `rdkit` molecule.
- **return_vocab** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will return an
identifier for each clique (ring, bond, bridged compounds, single).
(default: [False](https://docs.python.org/3/library/constants.html#False))

**Return type:**
: `(LongTensor, LongTensor, int)` if `return_vocab` is
[False](https://docs.python.org/3/library/constants.html#False), else `(LongTensor, LongTensor, int, LongTensor)`


**get_embeddings(*model: [Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)*, **args: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*, ***kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/embedding.html#get_embeddings)**
: Returns the output embeddings of all
[MessagePassing](../generated/torch_geometric.nn.conv.MessagePassing.html#torch_geometric.nn.conv.MessagePassing) layers in
`model`.


Internally, this method registers forward hooks on all
[MessagePassing](../generated/torch_geometric.nn.conv.MessagePassing.html#torch_geometric.nn.conv.MessagePassing) layers of a `model`,
and runs the forward pass of the `model` by calling
`model(*args, **kwargs)`.


**Parameters:**
: - **model** ([torch.nn.Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)) – The message passing model.
- ***args** – Arguments passed to the model.
- ****kwargs** (*optional*) – Additional keyword arguments passed to the model.

**Return type:**
: [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]


**get_embeddings_hetero(*model: [Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)*, *supported_models: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[Type](https://docs.python.org/3/library/typing.html#typing.Type)[[Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)]]] = None*, **args: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*, ***kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]][[source]](../_modules/torch_geometric/utils/embedding.html#get_embeddings_hetero)**
: Returns the output embeddings of all
[MessagePassing](../generated/torch_geometric.nn.conv.MessagePassing.html#torch_geometric.nn.conv.MessagePassing) layers in a heterogeneous
`model`, organized by edge type.


Internally, this method registers forward hooks on all modules that process
heterogeneous graphs in the model and runs the forward pass of the model.
For heterogeneous models, the output is a dictionary where each key is a
node type and each value is a list of embeddings from different layers.


**Parameters:**
: - **model** ([torch.nn.Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)) – The heterogeneous GNN model.
- **supported_models** (*List**[**Type**[*[torch.nn.Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)*]**]**, **optional*) – A list of
supported model classes. If not provided, defaults to
[HGTConv, HANConv, HeteroConv].
- ***args** – Arguments passed to the model.
- ****kwargs** (*optional*) – Additional keyword arguments passed to the model.

**Returns:**
: A dictionary mapping each node type to
a list of embeddings from different layers.

**Return type:**
: Dict[NodeType, List[Tensor]]


**trim_to_layer(*layer: [int](https://docs.python.org/3/library/functions.html#int)*, *num_sampled_nodes_per_hop: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]]*, *num_sampled_edges_per_hop: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)], [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [List](https://docs.python.org/3/library/typing.html#typing.List)[[int](https://docs.python.org/3/library/functions.html#int)]]]*, *x: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]*, *edge_index: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]] = None*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]], [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), SparseTensor]]], [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]]][[source]](../_modules/torch_geometric/utils/_trim_to_layer.html#trim_to_layer)**
: Trims the `edge_index` representation, node features `x` and
edge features `edge_attr` to a minimal-sized representation for the
current GNN layer `layer` in directed
[NeighborLoader](loader.html#torch_geometric.loader.NeighborLoader) scenarios.


This ensures that no computation is performed for nodes and edges that are
not included in the current GNN layer, thus avoiding unnecessary
computation within the GNN when performing neighborhood sampling.


**Parameters:**
: - **layer** ([int](https://docs.python.org/3/library/functions.html#int)) – The current GNN layer.
- **num_sampled_nodes_per_hop** (*List**[*[int](https://docs.python.org/3/library/functions.html#int)*] or **Dict**[**NodeType**, **List**[*[int](https://docs.python.org/3/library/functions.html#int)*]**]*) – The
number of sampled nodes per hop.
- **num_sampled_edges_per_hop** (*List**[*[int](https://docs.python.org/3/library/functions.html#int)*] or **Dict**[**EdgeType**, **List**[*[int](https://docs.python.org/3/library/functions.html#int)*]**]*) – The
number of sampled edges per hop.
- **x** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)* or **Dict**[**NodeType**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]*) – The homogeneous or
heterogeneous (hidden) node features.
- **edge_index** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)* or **Dict**[**EdgeType**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]*) – The
homogeneous or heterogeneous edge indices.
- **edge_attr** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)* or **Dict**[**EdgeType**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]**, **optional*) – The
homogeneous or heterogeneous (hidden) edge features.

**Return type:**
: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[`Union`[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]], `Union`[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], `Union`[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), `SparseTensor`]]], `Union`[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)], [None](https://docs.python.org/3/library/constants.html#None)]]


**get_ppr(*edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *alpha: [float](https://docs.python.org/3/library/functions.html#float) = 0.2*, *eps: [float](https://docs.python.org/3/library/functions.html#float) = 1e-05*, *target: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *num_nodes: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/utils/ppr.html#get_ppr)**
: Calculates the personalized PageRank (PPR) vector for all or a subset
of nodes using a variant of the [Andersen algorithm](https://mathweb.ucsd.edu/~fan/wp/localpartition.pdf).


**Parameters:**
: - **edge_index** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The indices of the graph.
- **alpha** ([float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – The alpha value of the PageRank algorithm.
(default: `0.2`)
- **eps** ([float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – The threshold for stopping the PPR calculation
(`edge_weight >= eps * out_degree`). (default: `1e-5`)
- **target** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The target nodes to compute PPR for.
If not given, calculates PPR vectors for all nodes.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_nodes** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of nodes. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor))


**train_test_split_edges(*data: [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)*, *val_ratio: [float](https://docs.python.org/3/library/functions.html#float) = 0.05*, *test_ratio: [float](https://docs.python.org/3/library/functions.html#float) = 0.1*) → [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)[[source]](../_modules/torch_geometric/utils/_train_test_split_edges.html#train_test_split_edges)**
: Splits the edges of a [torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) object
into positive and negative train/val/test edges.
As such, it will replace the `edge_index` attribute with
`train_pos_edge_index`, `train_pos_neg_adj_mask`,
`val_pos_edge_index`, `val_neg_edge_index` and
`test_pos_edge_index` attributes.
If `data` has edge features named `edge_attr`, then
`train_pos_edge_attr`, `val_pos_edge_attr` and
`test_pos_edge_attr` will be added as well.


> **Warning:** train_test_split_edges() is deprecated and
will be removed in a future release.
Use [torch_geometric.transforms.RandomLinkSplit](../generated/torch_geometric.transforms.RandomLinkSplit.html#torch_geometric.transforms.RandomLinkSplit) instead.


**Parameters:**
: - **data** ([Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)) – The data object.
- **val_ratio** ([float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – The ratio of positive validation edges.
(default: `0.05`)
- **test_ratio** ([float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – The ratio of positive test edges.
(default: `0.1`)

**Return type:**
: [torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)


**total_influence(*model: [Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)*, *data: [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)*, *max_hops: [int](https://docs.python.org/3/library/functions.html#int)*, *num_samples: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *normalize: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, *average: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, *device: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[device](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.device), [str](https://docs.python.org/3/library/stdtypes.html#str)] = 'cpu'*, *vectorize: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [float](https://docs.python.org/3/library/functions.html#float)][[source]](../_modules/torch_geometric/utils/influence.html#total_influence)**
: Compute Jacobian‑based influence aggregates for *multiple* seed nodes,
as introduced in the
[“Towards Quantifying Long-Range Interactions in Graph Machine Learning:
a Large Graph Dataset and a Measurement”](https://arxiv.org/abs/2503.09008) paper.
This measurement quantifies how a GNN model’s output at a node is
influenced by features of other nodes at increasing hop distances.


Specifically, for every sampled node $v$, this method


1. evaluates the **L1‑norm** of the Jacobian of the model output at
$v$ w.r.t. the node features of its *k*-hop induced sub‑graph;
2. sums these scores **per hop** to obtain the influence vector
$(I_{0}, I_{1}, \dots, I_{k})$;
3. optionally averages those vectors over all sampled nodes and
optionally normalises them by $I_{0}$.


Please refer to Section 4 of the paper for a more detailed definition.


**Parameters:**
: - **model** ([torch.nn.Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)) – A PyTorch Geometric‑compatible model with
forward signature `model(x, edge_index) -> Tensor`.
- **data** ([torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)) – Graph data object providing at least
`x` (node features) and `edge_index` (connectivity).
- **max_hops** ([int](https://docs.python.org/3/library/functions.html#int)) – Maximum hop distance $k$.
- **num_samples** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – Number of random seed nodes to evaluate.
If [None](https://docs.python.org/3/library/constants.html#None), all nodes are used. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **normalize** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If [True](https://docs.python.org/3/library/constants.html#True), normalize each hop‑wise
influence by the influence of hop 0. (default: [True](https://docs.python.org/3/library/constants.html#True))
- **average** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If [True](https://docs.python.org/3/library/constants.html#True), return the hop‑wise **mean**
over all seed nodes (shape `[k+1]`).
If [False](https://docs.python.org/3/library/constants.html#False), return the full influence matrix of shape
`[N, k+1]`. (default: [True](https://docs.python.org/3/library/constants.html#True))
- **device** ([torch.device](https://docs.pytorch.org/docs/main/tensor_attributes.html#torch.device)* or *[str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – Device on which to perform the
computation. (default: `"cpu"`)
- **vectorize** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – Forwarded to
[torch.autograd.functional.jacobian()](https://docs.pytorch.org/docs/main/generated/torch.autograd.functional.jacobian.html#torch.autograd.functional.jacobian).  Keeping this
[True](https://docs.python.org/3/library/constants.html#True) is often faster but increases memory usage.
(default: [True](https://docs.python.org/3/library/constants.html#True))

**Returns:**
: - **avg_influence** (*Tensor*):
shape `[k+1]` if `average=True`;
shape `[N, k+1]` otherwise.
- **R** (*float*): Influence‑weighted receptive‑field breadth
returned by `influence_weighted_receptive_field()`.

**Return type:**
: Tuple[Tensor, [float](https://docs.python.org/3/library/functions.html#float)]


**Example::**
: ```
>>> avg_I, R = total_influence(model, data, max_hops=3,
...                            num_samples=1000)
>>> avg_I
tensor([1.0000, 0.1273, 0.0142, 0.0019])
>>> R
0.216
```


