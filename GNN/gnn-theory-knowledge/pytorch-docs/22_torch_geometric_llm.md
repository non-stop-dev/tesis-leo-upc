| LargeGraphIndexer | For a dataset that consists of multiple subgraphs that are assumed to be part of a much larger graph, collate the values into a large graph store to save resources. |
| --- | --- |
| RAGQueryLoader | Loader meant for making RAG queries from a remote backend. |


***class *LargeGraphIndexer(*nodes: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*, *edges: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]]*, *node_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)]]] = None*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)]]] = None*)[[source]](../_modules/torch_geometric/llm/large_graph_indexer.html#LargeGraphIndexer)**
: For a dataset that consists of multiple subgraphs that are assumed to
be part of a much larger graph, collate the values into a large graph store
to save resources.


***classmethod *from_triplets(*triplets: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]]*, *pre_transform: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]], [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]]] = None*) → LargeGraphIndexer[[source]](../_modules/torch_geometric/llm/large_graph_indexer.html#LargeGraphIndexer.from_triplets)**
: Generate a new index from a series of triplets that represent edge
relations between nodes.
Formatted like (source_node, edge, dest_node).


**Parameters:**
: - **triplets** (*KnowledgeGraphLike*) – Series of triplets representing
knowledge graph relations. Example: [(“cats”, “eat”, dogs”)].
Note: Please ensure triplets are unique.
- **pre_transform** (*Optional**[**Callable**[**[**TripletLike**]**, **TripletLike**]**]*) – Optional preprocessing function to apply to triplets.
Defaults to None.

**Returns:**
: Index of unique nodes and edges.

**Return type:**
: LargeGraphIndexer


***classmethod *collate(*graphs: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[LargeGraphIndexer]*) → LargeGraphIndexer[[source]](../_modules/torch_geometric/llm/large_graph_indexer.html#LargeGraphIndexer.collate)**
: Combines a series of large graph indexes into a single large graph
index.


**Parameters:**
: **graphs** (*Iterable**[*LargeGraphIndexer*]*) – Indices to be
combined.

**Returns:**
: **Singular unique index for all nodes and edges**
: in input indices.

**Return type:**
: LargeGraphIndexer


**get_unique_node_features(*feature_name: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'pid'*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)][[source]](../_modules/torch_geometric/llm/large_graph_indexer.html#LargeGraphIndexer.get_unique_node_features)**
: Get all the unique values for a specific node attribute.


**Parameters:**
: **feature_name** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – Name of feature to get.
Defaults to NODE_PID.

**Returns:**
: List of unique values for the specified feature.

**Return type:**
: List[[str](https://docs.python.org/3/library/stdtypes.html#str)]


**add_node_feature(*new_feature_name: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *new_feature_vals: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *map_from_feature: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'pid'*) → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torch_geometric/llm/large_graph_indexer.html#LargeGraphIndexer.add_node_feature)**
: **Adds a new feature that corresponds to each unique node in**
: the graph.


**Parameters:**
: - **new_feature_name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – Name to call the new feature.
- **new_feature_vals** (*FeatureValueType*) – Values to map for that
new feature.
- **map_from_feature** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – Key of feature to map from.
Size must match the number of feature values.
Defaults to NODE_PID.

**Return type:**
: [None](https://docs.python.org/3/library/constants.html#None)


**get_node_features(*feature_name: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'pid'*, *pids: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)][[source]](../_modules/torch_geometric/llm/large_graph_indexer.html#LargeGraphIndexer.get_node_features)**
: **Get node feature values for a given set of unique node ids.**
: Returned values are not necessarily unique.


**Parameters:**
: - **feature_name** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – Name of feature to fetch. Defaults
to NODE_PID.
- **pids** (*Optional**[**Iterable**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]**]**, **optional*) – Node ids to fetch
for. Defaults to None, which fetches all nodes.

**Returns:**
: Node features corresponding to the specified ids.

**Return type:**
: List[Any]


**get_node_features_iter(*feature_name: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'pid'*, *pids: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*, *index_only: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Iterator](https://docs.python.org/3/library/typing.html#typing.Iterator)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)][[source]](../_modules/torch_geometric/llm/large_graph_indexer.html#LargeGraphIndexer.get_node_features_iter)**
: Iterator version of get_node_features. If index_only is True,
yields indices instead of values.


**Return type:**
: [Iterator](https://docs.python.org/3/library/typing.html#typing.Iterator)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)]


**get_unique_edge_features(*feature_name: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'e_pid'*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)][[source]](../_modules/torch_geometric/llm/large_graph_indexer.html#LargeGraphIndexer.get_unique_edge_features)**
: Get all the unique values for a specific edge attribute.


**Parameters:**
: **feature_name** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – Name of feature to get.
Defaults to EDGE_PID.

**Returns:**
: List of unique values for the specified feature.

**Return type:**
: List[[str](https://docs.python.org/3/library/stdtypes.html#str)]


**add_edge_feature(*new_feature_name: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *new_feature_vals: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *map_from_feature: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'e_pid'*) → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torch_geometric/llm/large_graph_indexer.html#LargeGraphIndexer.add_edge_feature)**
: Adds a new feature that corresponds to each unique edge in
the graph.


**Parameters:**
: - **new_feature_name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – Name to call the new feature.
- **new_feature_vals** (*FeatureValueType*) – Values to map for that new
feature.
- **map_from_feature** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – Key of feature to map from.
Size must match the number of feature values.
Defaults to EDGE_PID.

**Return type:**
: [None](https://docs.python.org/3/library/constants.html#None)


**get_edge_features(*feature_name: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'e_pid'*, *pids: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)][[source]](../_modules/torch_geometric/llm/large_graph_indexer.html#LargeGraphIndexer.get_edge_features)**
: **Get edge feature values for a given set of unique edge ids.**
: Returned values are not necessarily unique.


**Parameters:**
: - **feature_name** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – Name of feature to fetch.
Defaults to EDGE_PID.
- **pids** (*Optional**[**Iterable**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]**]**, **optional*) – Edge ids to fetch
for. Defaults to None, which fetches all edges.

**Returns:**
: Node features corresponding to the specified ids.

**Return type:**
: List[Any]


**get_edge_features_iter(*feature_name: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'e_pid'*, *pids: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)]]] = None*, *index_only: [bool](https://docs.python.org/3/library/functions.html#bool) = False*) → [Iterator](https://docs.python.org/3/library/typing.html#typing.Iterator)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)][[source]](../_modules/torch_geometric/llm/large_graph_indexer.html#LargeGraphIndexer.get_edge_features_iter)**
: Iterator version of get_edge_features. If index_only is True,
yields indices instead of values.


**Return type:**
: [Iterator](https://docs.python.org/3/library/typing.html#typing.Iterator)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)]


**to_data(*node_feature_name: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *edge_feature_name: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*) → [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)[[source]](../_modules/torch_geometric/llm/large_graph_indexer.html#LargeGraphIndexer.to_data)**
: **Return a Data object containing all the specified node and**
: edge features and the graph.


**Parameters:**
: - **node_feature_name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – Feature to use for nodes
- **edge_feature_name** (*Optional**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]**, **optional*) – Feature to use for
edges. Defaults to None.

**Returns:**
: **Data object containing the specified node and**
: edge features and the graph.

**Return type:**
: [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)


***class *RAGQueryLoader(*graph_data: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[RAGFeatureStore, RAGGraphStore]*, *subgraph_filter: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), [Any](https://docs.python.org/3/library/typing.html#typing.Any)], [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)]] = None*, *augment_query: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *vector_retriever: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[VectorRetriever] = None*, *config: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]] = None*)[[source]](../_modules/torch_geometric/llm/rag_loader.html#RAGQueryLoader)**
: Loader meant for making RAG queries from a remote backend.


***property *config**
: Get the config for the RAGQueryLoader.


**query(*query: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)[[source]](../_modules/torch_geometric/llm/rag_loader.html#RAGQueryLoader.query)**
: Retrieve a subgraph associated with the query with all its feature
attributes.


**Return type:**
: [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)


## Models


| SentenceTransformer | A wrapper around a Sentence-Transformer from HuggingFace. |
| --- | --- |
| VisionTransformer | A wrapper around a Vision-Transformer from HuggingFace. |
| LLM | A wrapper around a Large Language Model (LLM) from HuggingFace. |
| LLMJudge | Uses NIMs to score a triple of (question, model_pred, correct_answer) This whole class is an adaptation of Gilberto's work for PyG. |
| TXT2KG | A class to convert text data into a Knowledge Graph (KG) format. |
| GRetriever | The G-Retriever model from the"G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering"paper. |
| MoleculeGPT | The MoleculeGPT model from the"MoleculeGPT: Instruction Following Large Language Models for Molecular Property Prediction"paper. |
| GLEM | This GNN+LM co-training model is based on GLEM from the"Learning on Large-scale Text-attributed Graphs via Variational Inference"paper. |
| ProteinMPNN | The ProteinMPNN model from the"Robust deep learning--based protein sequence design using ProteinMPNN"paper. |
| GITMol | The GITMol model from the"GIT-Mol: A Multi-modal Large Language Model for Molecular Science with Graph, Image, and Text"paper. |


## Utils


| KNNRAGFeatureStore | A feature store that uses a KNN-based retrieval. |
| --- | --- |
| NeighborSamplingRAGGraphStore | Neighbor sampling based graph-store to store & retrieve graph data. |
| DocumentRetriever | Retrieve documents from a vector database. |


