`torch_geometric.contrib` is a staging area for early stage experimental code.
Modules might be moved to the main library in the future.


> **Warning:** This module contains experimental code, which is not guaranteed to be stable.


## Convolutional Layers


## Models


| PRBCDAttack | The Projected Randomized Block Coordinate Descent (PRBCD) adversarial attack from theRobustness of Graph Neural Networks at Scalepaper. |
| --- | --- |
| GRBCDAttack | The Greedy Randomized Block Coordinate Descent (GRBCD) adversarial attack from theRobustness of Graph Neural Networks at Scalepaper. |


***class *PRBCDAttack(*model: [Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)*, *block_size: [int](https://docs.python.org/3/library/functions.html#int)*, *epochs: [int](https://docs.python.org/3/library/functions.html#int) = 125*, *epochs_resampling: [int](https://docs.python.org/3/library/functions.html#int) = 100*, *loss: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]] = 'prob_margin'*, *metric: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]] = None*, *lr: [float](https://docs.python.org/3/library/functions.html#float) = 1000*, *is_undirected: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, *log: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, ***kwargs*)[[source]](../_modules/torch_geometric/contrib/nn/models/rbcd_attack.html#PRBCDAttack)**
: The Projected Randomized Block Coordinate Descent (PRBCD) adversarial
attack from the [Robustness of Graph Neural Networks at Scale](https://www.cs.cit.tum.de/daml/robustness-of-gnns-at-scale) paper.


This attack uses an efficient gradient based approach that (during the
attack) relaxes the discrete entries in the adjacency matrix
$\{0, 1\}$ to $[0, 1]$ and solely perturbs the adjacency matrix
(no feature perturbations). Thus, this attack supports all models that can
handle weighted graphs that are differentiable w.r.t. these edge weights,
*e.g.*, [GCNConv](../generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv) or
[GraphConv](../generated/torch_geometric.nn.conv.GraphConv.html#torch_geometric.nn.conv.GraphConv). For non-differentiable models
you might need modifications, e.g., see example for
[GATConv](../generated/torch_geometric.nn.conv.GATConv.html#torch_geometric.nn.conv.GATConv).


The memory overhead is driven by the additional edges (at most
`block_size`). For scalability reasons, the block is drawn with
replacement and then the index is made unique. Thus, the actual block size
is typically slightly smaller than specified.


This attack can be used for both global and local attacks as well as
test-time attacks (evasion) and training-time attacks (poisoning). Please
see the provided examples.


This attack is designed with a focus on node- or graph-classification,
however, to adapt to other tasks you most likely only need to provide an
appropriate loss and model. However, we currently do not support batching
out of the box (sampling needs to be adapted).


> **Note:** For examples of using the PRBCD Attack, see
[examples/contrib/rbcd_attack.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/contrib/rbcd_attack.py)
for a test time attack (evasion) or
[examples/contrib/rbcd_attack_poisoning.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/contrib/rbcd_attack_poisoning.py)
for a training time (poisoning) attack.


**Parameters:**
: - **model** ([torch.nn.Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)) – The GNN module to assess.
- **block_size** ([int](https://docs.python.org/3/library/functions.html#int)) – Number of randomly selected elements in the
adjacency matrix to consider.
- **epochs** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – Number of epochs (aborts early if
`mode='greedy'` and budget is satisfied) (default: `125`)
- **epochs_resampling** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – Number of epochs to resample the
random block. (default: obj:100)
- **loss** ([str](https://docs.python.org/3/library/stdtypes.html#str)* or **callable**, **optional*) – A loss to quantify the “strength” of
an attack. Note that this function must match the output format of
`model`. By default, it is assumed that the task is
classification and that the model returns raw predictions (*i.e.*,
no output activation) or uses `logsoftmax`. Moreover, and the
number of predictions should match the number of labels passed to
attack. Either pass a callable or one of: `'masked'`,
`'margin'`, `'prob_margin'`, `'tanh_margin'`.
(default: `'prob_margin'`)
- **metric** (*callable**, **optional*) – Second (potentially
non-differentiable) loss for monitoring or early stopping (if
`mode='greedy'`). (default: same as `loss`)
- **lr** ([float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – Learning rate for updating edge weights.
Additionally, it is heuristically corrected for `block_size`,
budget (see attack) and graph size. (default: `1_000`)
- **is_undirected** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If [True](https://docs.python.org/3/library/constants.html#True) the graph is
assumed to be undirected. (default: [True](https://docs.python.org/3/library/constants.html#True))
- **log** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [False](https://docs.python.org/3/library/constants.html#False), will not log any learning
progress. (default: [True](https://docs.python.org/3/library/constants.html#True))


**coeffs* = {'eps': 1e-07, 'max_final_samples': 20, 'max_trials_sampling': 20, 'with_early_stopping': True}***
:


**attack(*x: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *labels: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *budget: [int](https://docs.python.org/3/library/functions.html#int)*, *idx_attack: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)] = None*, ***kwargs*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)][[source]](../_modules/torch_geometric/contrib/nn/models/rbcd_attack.html#PRBCDAttack.attack)**
: Attack the predictions for the provided model and graph.


A subset of predictions may be specified with `idx_attack`. The
attack is allowed to flip (i.e. add or delete) `budget` edges and
will return the strongest perturbation it can find. It returns both the
resulting perturbed `edge_index` as well as the perturbations.


**Parameters:**
: - **x** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The node feature matrix.
- **edge_index** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The edge indices.
- **labels** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The labels.
- **budget** ([int](https://docs.python.org/3/library/functions.html#int)) – The number of allowed perturbations (i.e.
number of edges that are flipped at most).
- **idx_attack** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – Filter for predictions/labels.
Shape and type must match that it can index `labels`
and the model’s predictions.
- ****kwargs** (*optional*) – Additional arguments passed to the GNN module.

**Return type:**
: ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor))


***class *GRBCDAttack(*model: [Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)*, *block_size: [int](https://docs.python.org/3/library/functions.html#int)*, *epochs: [int](https://docs.python.org/3/library/functions.html#int) = 125*, *loss: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor), [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]], [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]]] = 'masked'*, *is_undirected: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, *log: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, ***kwargs*)[[source]](../_modules/torch_geometric/contrib/nn/models/rbcd_attack.html#GRBCDAttack)**
: The Greedy Randomized Block Coordinate Descent (GRBCD) adversarial
attack from the [Robustness of Graph Neural Networks at Scale](https://www.cs.cit.tum.de/daml/robustness-of-gnns-at-scale) paper.


GRBCD shares most of the properties and requirements with
PRBCDAttack. It also uses an efficient gradient based approach.
However, it greedily flips edges based on the gradient towards the
adjacency matrix.


> **Note:** For examples of using the GRBCD Attack, see
[examples/contrib/rbcd_attack.py](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/contrib/rbcd_attack.py)
for a test time attack (evasion).


**Parameters:**
: - **model** ([torch.nn.Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)) – The GNN module to assess.
- **block_size** ([int](https://docs.python.org/3/library/functions.html#int)) – Number of randomly selected elements in the
adjacency matrix to consider.
- **epochs** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – Number of epochs (aborts early if
`mode='greedy'` and budget is satisfied) (default: `125`)
- **loss** ([str](https://docs.python.org/3/library/stdtypes.html#str)* or **callable**, **optional*) – A loss to quantify the “strength” of
an attack. Note that this function must match the output format of
`model`. By default, it is assumed that the task is
classification and that the model returns raw predictions (*i.e.*,
no output activation) or uses `logsoftmax`. Moreover, and the
number of predictions should match the number of labels passed to
`attack`. Either pass Callable or one of: `'masked'`,
`'margin'`, `'prob_margin'`, `'tanh_margin'`.
(default: `'masked'`)
- **is_undirected** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If [True](https://docs.python.org/3/library/constants.html#True) the graph is
assumed to be undirected. (default: [True](https://docs.python.org/3/library/constants.html#True))
- **log** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [False](https://docs.python.org/3/library/constants.html#False), will not log any learning
progress. (default: [True](https://docs.python.org/3/library/constants.html#True))


**coeffs* = {'eps': 1e-07, 'max_trials_sampling': 20}***
:


## Datasets


## Transforms


## Explainer


| PGMExplainer | The PGMExplainer model from the"PGMExplainer: Probabilistic Graphical Model Explanations  for Graph Neural Networks"paper. |
| --- | --- |


***class *PGMExplainer(*feature_index: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[List](https://docs.python.org/3/library/typing.html#typing.List)] = None*, *perturbation_mode: [str](https://docs.python.org/3/library/stdtypes.html#str) = 'randint'*, *perturbations_is_positive_only: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *is_perturbation_scaled: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *num_samples: [int](https://docs.python.org/3/library/functions.html#int) = 100*, *max_subgraph_size: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*, *significance_threshold: [float](https://docs.python.org/3/library/functions.html#float) = 0.05*, *pred_threshold: [float](https://docs.python.org/3/library/functions.html#float) = 0.1*)[[source]](../_modules/torch_geometric/contrib/explain/pgm_explainer.html#PGMExplainer)**
: The PGMExplainer model from the [“PGMExplainer: Probabilistic
Graphical Model Explanations  for Graph Neural Networks”](https://arxiv.org/abs/1903.03894) paper.


The generated [Explanation](explain.html#torch_geometric.explain.Explanation) provides a
`node_mask` and a `pgm_stats` tensor, which stores the
$p$-values of each node as calculated by the Chi-squared test.


**Parameters:**
: - **feature_index** (*List*) – The indices of the perturbed features. If set
to [None](https://docs.python.org/3/library/constants.html#None), all features are perturbed. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **perturb_mode** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The method to generate the variations in
features. One of `"randint"`, `"mean"`, `"zero"`,
`"max"` or `"uniform"`. (default: `"randint"`)
- **perturbations_is_positive_only** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True),
restrict perturbed values to be positive. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **is_perturbation_scaled** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will
normalize the range of the perturbed features.
(default: [False](https://docs.python.org/3/library/constants.html#False))
- **num_samples** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of samples of perturbations
used to test the significance of nodes to the prediction.
(default: `100`)
- **max_subgraph_size** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The maximum number of neighbors to
consider for the explanation. (default: [None](https://docs.python.org/3/library/constants.html#None))
- **significance_threshold** ([float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – The statistical threshold
($p$-value) for which a node is considered to have an effect
on the prediction. (default: `0.05`)
- **pred_threshold** ([float](https://docs.python.org/3/library/functions.html#float)*, **optional*) – The buffer value (in range
`[0, 1]`) to consider the output from a perturbed data to be
different from the original. (default: `0.1`)


**forward(*model: [Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)*, *x: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *edge_index: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, ***, *target: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *index: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[int](https://docs.python.org/3/library/functions.html#int), [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]] = None*, ***kwargs*) → [Explanation](explain.html#torch_geometric.explain.Explanation)[[source]](../_modules/torch_geometric/contrib/explain/pgm_explainer.html#PGMExplainer.forward)**
: Computes the explanation.


**Parameters:**
: - **model** ([torch.nn.Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)) – The model to explain.
- **x** (*Union**[*[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **Dict**[**NodeType**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]**]*) – The input
node features of a homogeneous or heterogeneous graph.
- **edge_index** (*Union**[*[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **Dict**[**NodeType**, *[torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*]**]*) – The
input edge indices of a homogeneous or heterogeneous graph.
- **target** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – The target of the model.
- **index** (*Union**[*[int](https://docs.python.org/3/library/functions.html#int)*, **Tensor**]**, **optional*) – The index of the model
output to explain. Can be a single index or a tensor of
indices. (default: [None](https://docs.python.org/3/library/constants.html#None))
- ****kwargs** (*optional*) – Additional keyword arguments passed to
`model`.

**Return type:**
: [Explanation](explain.html#torch_geometric.explain.Explanation)


**supports() → [bool](https://docs.python.org/3/library/functions.html#bool)[[source]](../_modules/torch_geometric/contrib/explain/pgm_explainer.html#PGMExplainer.supports)**
: Checks if the explainer supports the user-defined settings provided
in `self.explainer_config`, `self.model_config`.


**Return type:**
: [bool](https://docs.python.org/3/library/functions.html#bool)


