> **Warning:** This module is in active development and may not be stable.
Access requires installing PyG from master.


## Philosophy


This module provides a set of tools to explain the predictions of a PyG model or to explain the underlying phenomenon of a dataset (see the [“GraphFramEx: Towards Systematic Evaluation of Explainability Methods for Graph Neural Networks”](https://arxiv.org/abs/2206.09677) paper for more details).


We represent explanations using the torch_geometric.explain.Explanation class, which is a [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) object containing masks for the nodes, edges, features and any attributes of the data.


The torch_geometric.explain.Explainer class is designed to handle all explainability parameters (see the torch_geometric.explain.config.ExplainerConfig class for more details):


- which algorithm from the `torch_geometric.explain.algorithm` module to use (*e.g.*, [GNNExplainer](../generated/torch_geometric.explain.algorithm.GNNExplainer.html#torch_geometric.explain.algorithm.GNNExplainer))
- the type of explanation to compute (*e.g.*, `explanation_type="phenomenon"` or `explanation_type="model"`)
- the different type of masks for node and edges (*e.g.*, `mask="object"` or `mask="attributes"`)
- any postprocessing of the masks (*e.g.*, `threshold_type="topk"` or `threshold_type="hard"`)


This class allows the user to easily compare different explainability methods and to easily switch between different types of masks, while making sure the high-level framework stays the same.


## Explainer


***class *Explainer(*model: [Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)*, *algorithm: [ExplainerAlgorithm](../generated/torch_geometric.explain.algorithm.ExplainerAlgorithm.html#torch_geometric.explain.algorithm.ExplainerAlgorithm)*, *explanation_type: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[ExplanationType, [str](https://docs.python.org/3/library/stdtypes.html#str)]*, *model_config: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[ModelConfig, [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]*, *node_mask_type: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[MaskType, [str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*, *edge_mask_type: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[MaskType, [str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*, *threshold_config: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[ThresholdConfig] = None*)[[source]](../_modules/torch_geometric/explain/explainer.html#Explainer)**
: Bases: [object](https://docs.python.org/3/library/functions.html#object)


An explainer class for instance-level explanations of Graph Neural
Networks.


**Parameters:**
: - **model** ([torch.nn.Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)) – The model to explain.
- **algorithm** ([ExplainerAlgorithm](../generated/torch_geometric.explain.algorithm.ExplainerAlgorithm.html#torch_geometric.explain.algorithm.ExplainerAlgorithm)) – The explanation algorithm.
- **explanation_type** (*ExplanationType** or *[str](https://docs.python.org/3/library/stdtypes.html#str)) –
The type of explanation to
compute. The possible values are:


> - `"model"`: Explains the model prediction.
> - `"phenomenon"`: Explains the phenomenon that the model
> is trying to predict.


In practice, this means that the explanation algorithm will either
compute their losses with respect to the model output
(`"model"`) or the target output (`"phenomenon"`).
- **model_config** (ModelConfig) – The model configuration.
See ModelConfig for
available options. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **node_mask_type** (*MaskType** or *[str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) –
The type of mask to apply
on nodes. The possible values are (default: [None](https://docs.python.org/3/library/constants.html#None)):


> - [None](https://docs.python.org/3/library/constants.html#None): Will not apply any mask on nodes.
> - `"object"`: Will mask each node.
> - `"common_attributes"`: Will mask each feature.
> - `"attributes"`: Will mask each feature across all nodes.
- **edge_mask_type** (*MaskType** or *[str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The type of mask to apply
on edges. Has the sample possible values as `node_mask_type`.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **threshold_config** (ThresholdConfig*, **optional*) – The threshold
configuration.
See ThresholdConfig for
available options. (default: [None](https://docs.python.org/3/library/constants.html#None))


**get_prediction(**args*, ***kwargs*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/explain/explainer.html#Explainer.get_prediction)**
: Returns the prediction of the model on the input graph.


If the model mode is `"regression"`, the prediction is returned as
a scalar value.
If the model mode is `"multiclass_classification"` or
`"binary_classification"`, the prediction is returned as the
predicted class label.


**Parameters:**
: - ***args** – Arguments passed to the model.
- ****kwargs** (*optional*) – Additional keyword arguments passed to the
model.

**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


**get_masked_prediction(*x: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]*, *edge_index: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]*, *node_mask: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]] = None*, *edge_mask: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]] = None*, ***kwargs*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/explain/explainer.html#Explainer.get_masked_prediction)**
: Returns the prediction of the model on the input graph with node
and edge masks applied.


**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


**__call__(*x: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]*, *edge_index: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str), [str](https://docs.python.org/3/library/stdtypes.html#str)], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]*, ***, *target: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *index: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]] = None*, ***kwargs*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[Explanation, HeteroExplanation][[source]](../_modules/torch_geometric/explain/explainer.html#Explainer.__call__)**
: Computes the explanation of the GNN for the given inputs and
target.


> **Note:** If you get an error message like “Trying to backward through the
graph a second time”, make sure that the target you provided
was computed with `torch.no_grad()`.


**Parameters:**
: - **x** (*Union**[*[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **Dict**[**NodeType**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]**]*) – The input
node features of a homogeneous or heterogeneous graph.
- **edge_index** (*Union**[*[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **Dict**[**NodeType**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]**]*) – The
input edge indices of a homogeneous or heterogeneous graph.
- **target** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The target of the model.
If the explanation type is `"phenomenon"`, the target has
to be provided.
If the explanation type is `"model"`, the target should be
set to [None](https://docs.python.org/3/library/constants.html#None) and will get automatically inferred. For
classification tasks, the target needs to contain the class
labels. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **index** (*Union**[*[int](https://docs.python.org/3/library/functions.html#int)*, **Tensor**]**, **optional*) – The indices in the
first-dimension of the model output to explain.
Can be a single index or a tensor of indices.
If set to [None](https://docs.python.org/3/library/constants.html#None), all model outputs will be explained.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- ****kwargs** – additional arguments to pass to the GNN.

**Return type:**
: `Union`[Explanation, HeteroExplanation]


**get_target(*prediction: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/explain/explainer.html#Explainer.get_target)**
: Returns the target of the model from a given prediction.


If the model mode is of type `"regression"`, the prediction is
returned as it is.
If the model mode is of type `"multiclass_classification"` or
`"binary_classification"`, the prediction is returned as the
predicted class label.


**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


***class *ExplainerConfig(*explanation_type: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[ExplanationType, [str](https://docs.python.org/3/library/stdtypes.html#str)]*, *node_mask_type: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[MaskType, [str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*, *edge_mask_type: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[MaskType, [str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*)[[source]](../_modules/torch_geometric/explain/config.html#ExplainerConfig)**
: Configuration class to store and validate high level explanation
parameters.


**Parameters:**
: - **explanation_type** (*ExplanationType** or *[str](https://docs.python.org/3/library/stdtypes.html#str)) –
The type of explanation to
compute. The possible values are:


> - `"model"`: Explains the model prediction.
> - `"phenomenon"`: Explains the phenomenon that the model
> is trying to predict.


In practice, this means that the explanation algorithm will either
compute their losses with respect to the model output
(`"model"`) or the target output (`"phenomenon"`).
- **node_mask_type** (*MaskType** or *[str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) –
The type of mask to apply
on nodes. The possible values are (default: [None](https://docs.python.org/3/library/constants.html#None)):


> - [None](https://docs.python.org/3/library/constants.html#None): Will not apply any mask on nodes.
> - `"object"`: Will mask each node.
> - `"common_attributes"`: Will mask each feature.
> - `"attributes"`: Will mask each feature across all nodes.
- **edge_mask_type** (*MaskType** or *[str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The type of mask to apply
on edges. Has the sample possible values as `node_mask_type`.
(default: [None](https://docs.python.org/3/library/constants.html#None))


***class *ModelConfig(*mode: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[ModelMode, [str](https://docs.python.org/3/library/stdtypes.html#str)]*, *task_level: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[ModelTaskLevel, [str](https://docs.python.org/3/library/stdtypes.html#str)]*, *return_type: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[ModelReturnType, [str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*)[[source]](../_modules/torch_geometric/explain/config.html#ModelConfig)**
: Configuration class to store model parameters.


**Parameters:**
: - **mode** (*ModelMode** or *[str](https://docs.python.org/3/library/stdtypes.html#str)) –
The mode of the model. The possible values
are:


> - `"binary_classification"`: A binary classification
> model.
> - `"multiclass_classification"`: A multiclass
> classification model.
> - `"regression"`: A regression model.
- **task_level** (*ModelTaskLevel** or *[str](https://docs.python.org/3/library/stdtypes.html#str)) –
The task-level of the model.
The possible values are:


> - `"node"`: A node-level prediction model.
> - `"edge"`: An edge-level prediction model.
> - `"graph"`: A graph-level prediction model.
- **return_type** (*ModelReturnType** or *[str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) –
The return type of the
model. The possible values are (default: [None](https://docs.python.org/3/library/constants.html#None)):


> - `"raw"`: The model returns raw values.
> - `"probs"`: The model returns probabilities.
> - `"log_probs"`: The model returns log-probabilities.


***class *ThresholdConfig(*threshold_type: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[ThresholdType, [str](https://docs.python.org/3/library/stdtypes.html#str)]*, *value: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[float](https://docs.python.org/3/library/functions.html#float), [int](https://docs.python.org/3/library/functions.html#int)]*)[[source]](../_modules/torch_geometric/explain/config.html#ThresholdConfig)**
: Configuration class to store and validate threshold parameters.


**Parameters:**
: - **threshold_type** (*ThresholdType** or *[str](https://docs.python.org/3/library/stdtypes.html#str)) –
The type of threshold to apply.
The possible values are:


> - [None](https://docs.python.org/3/library/constants.html#None): No threshold is applied.
> - `"hard"`: A hard threshold is applied to each mask.
> The elements of the mask with a value below the `value`
> are set to `0`, the others are set to `1`.
> - `"topk"`: A soft threshold is applied to each mask.
> The top obj:value elements of each mask are kept, the
> others are set to `0`.
> - `"topk_hard"`: Same as `"topk"` but values are set
> to `1` for all elements which are kept.
- **value** ([int](https://docs.python.org/3/library/functions.html#int)* or *[float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – The value to use when thresholding.
(default: [None](https://docs.python.org/3/library/constants.html#None))


## Explanations


***class *Explanation(*x: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *edge_index: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *edge_attr: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *y: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [int](https://docs.python.org/3/library/functions.html#int), [float](https://docs.python.org/3/library/functions.html#float)]] = None*, *pos: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, *time: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, ***kwargs*)[[source]](../_modules/torch_geometric/explain/explanation.html#Explanation)**
: Bases: [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data), `ExplanationMixin`


Holds all the obtained explanations of a homogeneous graph.


The explanation object is a [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) object and
can hold node attributions and edge attributions.
It can also hold the original graph if needed.


**Parameters:**
: - **node_mask** (*Tensor**, **optional*) – Node-level mask with shape
`[num_nodes, 1]`, `[1, num_features]` or
`[num_nodes, num_features]`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **edge_mask** (*Tensor**, **optional*) – Edge-level mask with shape
`[num_edges]`. (default: [None](https://docs.python.org/3/library/constants.html#None))
- ****kwargs** (*optional*) – Additional attributes.


**validate(*raise_on_error: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → [bool](https://docs.python.org/3/library/functions.html#bool)[[source]](../_modules/torch_geometric/explain/explanation.html#Explanation.validate)**
: Validates the correctness of the Explanation object.


**Return type:**
: [bool](https://docs.python.org/3/library/functions.html#bool)


**get_explanation_subgraph() → Explanation[[source]](../_modules/torch_geometric/explain/explanation.html#Explanation.get_explanation_subgraph)**
: Returns the induced subgraph, in which all nodes and edges with
zero attribution are masked out.


**Return type:**
: Explanation


**get_complement_subgraph() → Explanation[[source]](../_modules/torch_geometric/explain/explanation.html#Explanation.get_complement_subgraph)**
: Returns the induced subgraph, in which all nodes and edges with any
attribution are masked out.


**Return type:**
: Explanation


**visualize_feature_importance(*path: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *feat_labels: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*, *top_k: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*)[[source]](../_modules/torch_geometric/explain/explanation.html#Explanation.visualize_feature_importance)**
: Creates a bar plot of the node feature importances by summing up
the node mask across all nodes.


**Parameters:**
: - **path** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The path to where the plot is saved.
If set to [None](https://docs.python.org/3/library/constants.html#None), will visualize the plot on-the-fly.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **feat_labels** (*List**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]**, **optional*) – The labels of features.
(default [None](https://docs.python.org/3/library/constants.html#None))
- **top_k** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – Top k features to plot. If [None](https://docs.python.org/3/library/constants.html#None)
plots all features. (default: [None](https://docs.python.org/3/library/constants.html#None))


**visualize_graph(*path: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *backend: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *node_labels: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*) → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torch_geometric/explain/explanation.html#Explanation.visualize_graph)**
: Visualizes the explanation graph with edge opacity corresponding to
edge importance.


**Parameters:**
: - **path** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The path to where the plot is saved.
If set to [None](https://docs.python.org/3/library/constants.html#None), will visualize the plot on-the-fly.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **backend** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The graph drawing backend to use for
visualization (`"graphviz"`, `"networkx"`).
If set to [None](https://docs.python.org/3/library/constants.html#None), will use the most appropriate
visualization backend based on available system packages.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **node_labels** ([list](https://docs.python.org/3/library/stdtypes.html#list)*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]**, **optional*) – The labels/IDs of nodes.
(default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: [None](https://docs.python.org/3/library/constants.html#None)


***class *HeteroExplanation(*_mapping: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]] = None*, ***kwargs*)[[source]](../_modules/torch_geometric/explain/explanation.html#HeteroExplanation)**
: Bases: [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData), `ExplanationMixin`


Holds all the obtained explanations of a heterogeneous graph.


The explanation object is a [HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData) object
and can hold node attributions and edge attributions.
It can also hold the original graph if needed.


**validate(*raise_on_error: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → [bool](https://docs.python.org/3/library/functions.html#bool)[[source]](../_modules/torch_geometric/explain/explanation.html#HeteroExplanation.validate)**
: Validates the correctness of the Explanation object.


**Return type:**
: [bool](https://docs.python.org/3/library/functions.html#bool)


**get_explanation_subgraph() → HeteroExplanation[[source]](../_modules/torch_geometric/explain/explanation.html#HeteroExplanation.get_explanation_subgraph)**
: Returns the induced subgraph, in which all nodes and edges with
zero attribution are masked out.


**Return type:**
: HeteroExplanation


**get_complement_subgraph() → HeteroExplanation[[source]](../_modules/torch_geometric/explain/explanation.html#HeteroExplanation.get_complement_subgraph)**
: Returns the induced subgraph, in which all nodes and edges with any
attribution are masked out.


**Return type:**
: HeteroExplanation


**visualize_feature_importance(*path: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *feat_labels: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)]]] = None*, *top_k: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*)[[source]](../_modules/torch_geometric/explain/explanation.html#HeteroExplanation.visualize_feature_importance)**
: Creates a bar plot of the node feature importances by summing up
node masks across all nodes for each node type.


**Parameters:**
: - **path** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The path to where the plot is saved.
If set to [None](https://docs.python.org/3/library/constants.html#None), will visualize the plot on-the-fly.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **feat_labels** (*Dict**[**NodeType**, **List**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]**]**, **optional*) – The labels of
features for each node type. (default [None](https://docs.python.org/3/library/constants.html#None))
- **top_k** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – Top k features to plot. If [None](https://docs.python.org/3/library/constants.html#None)
plots all features. (default: [None](https://docs.python.org/3/library/constants.html#None))


**visualize_graph(*path: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *node_labels: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)]]] = None*, *node_size_range: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)] = (50, 500)*, *node_opacity_range: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)] = (0.2, 1.0)*, *edge_width_range: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)] = (0.1, 2.0)*, *edge_opacity_range: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)] = (0.2, 1.0)*) → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torch_geometric/explain/explanation.html#HeteroExplanation.visualize_graph)**
: Visualizes the explanation subgraph using networkx, with edge
opacity corresponding to edge importance and node colors
corresponding to node types.


**Parameters:**
: - **path** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The path to where the plot is saved.
If set to [None](https://docs.python.org/3/library/constants.html#None), will visualize the plot on-the-fly.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **node_labels** (*Dict**[**NodeType**, **List**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]**]**, **optional*) – The display
names of nodes for each node type that will be shown in the
visualization. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **node_size_range** (*Tuple**[*[float](https://docs.python.org/3/library/functions.html#float)*, *[float](https://docs.python.org/3/library/functions.html#float)*]**, **optional*) – The minimum and
maximum node size in the visualization.
(default: `(50, 500)`)
- **node_opacity_range** (*Tuple**[*[float](https://docs.python.org/3/library/functions.html#float)*, *[float](https://docs.python.org/3/library/functions.html#float)*]**, **optional*) – The minimum and
maximum node opacity in the visualization.
(default: `(0.2, 1.0)`)
- **edge_width_range** (*Tuple**[*[float](https://docs.python.org/3/library/functions.html#float)*, *[float](https://docs.python.org/3/library/functions.html#float)*]**, **optional*) – The minimum and
maximum edge width in the visualization.
(default: `(0.1, 2.0)`)
- **edge_opacity_range** (*Tuple**[*[float](https://docs.python.org/3/library/functions.html#float)*, *[float](https://docs.python.org/3/library/functions.html#float)*]**, **optional*) – The minimum and
maximum edge opacity in the visualization.
(default: `(0.2, 1.0)`)

**Return type:**
: [None](https://docs.python.org/3/library/constants.html#None)


## Explainer Algorithms


| ExplainerAlgorithm | An abstract base class for implementing explainer algorithms. |
| --- | --- |
| DummyExplainer | A dummy explainer that returns random explanations (useful for testing purposes). |
| GNNExplainer | The GNN-Explainer model from the"GNNExplainer: Generating Explanations for Graph Neural Networks"paper for identifying compact subgraph structures and node features that play a crucial role in the predictions made by a GNN. |
| CaptumExplainer | ACaptum-based explainer for identifying compact subgraph structures and node features that play a crucial role in the predictions made by a GNN. |
| PGExplainer | The PGExplainer model from the"Parameterized Explainer for Graph Neural Network"paper. |
| AttentionExplainer | An explainer that uses the attention coefficients produced by an attention-based GNN (e.g.,GATConv,GATv2Conv, orTransformerConv) as edge explanation. |
| GraphMaskExplainer | The GraphMask-Explainer model from the"Interpreting Graph Neural Networks for NLP With Differentiable Edge Masking"paper for identifying layer-wise compact subgraph structures and node features that play a crucial role in the predictions made by a GNN. |


## Explanation Metrics


The quality of an explanation can be judged by a variety of different methods.
PyG supports the following metrics out-of-the-box:


| groundtruth_metrics | Compares and evaluates an explanation mask with the ground-truth explanation mask. |
| --- | --- |
| fidelity | Evaluates the fidelity of anExplainergiven anExplanation, as described in the"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for Graph Neural Networks"paper. |
| characterization_score | Returns the componentwise characterization score as described in the"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for Graph Neural Networks"paper. |
| fidelity_curve_auc | Returns the AUC for the fidelity curve as described in the"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for Graph Neural Networks"paper. |
| unfaithfulness | Evaluates how faithful anExplanationis to an underlying GNN predictor, as described in the"Evaluating Explainability for Graph Neural Networks"paper. |


