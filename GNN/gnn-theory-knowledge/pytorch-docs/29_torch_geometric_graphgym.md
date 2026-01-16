## Workflow and Register Modules


| load_ckpt | Loads the model checkpoint at a given epoch. |
| --- | --- |
| save_ckpt | Saves the model checkpoint at a given epoch. |
| remove_ckpt | Removes the model checkpoint at a given epoch. |
| clean_ckpt | Removes all but the last model checkpoint. |
| parse_args | Parses the command line arguments. |
| cfg |  |
| set_cfg | This function sets the default config value. |
| load_cfg | Load configurations from file system and command line. |
| dump_cfg | Dumps the config to the output directory specified incfg.out_dir. |
| set_run_dir | Create the directory for each random seed experiment run. |
| set_out_dir | Create the directory for full experiment run. |
| get_fname | Extract filename from file name path. |
| init_weights | Performs weight initialization. |
| create_loader | Create data loader object. |
| set_printing | Set up printing options. |
| create_logger | Create logger for the experiment. |
| compute_loss | Compute loss and prediction score. |
| create_model | Create model for graph machine learning. |
| create_optimizer | Creates a config-driven optimizer. |
| create_scheduler | Creates a config-driven learning rate scheduler. |
| train | Trains a GraphGym model using PyTorch Lightning. |
| register_base | Base function for registering a module in GraphGym. |
| register_act | Registers an activation function in GraphGym. |
| register_node_encoder | Registers a node feature encoder in GraphGym. |
| register_edge_encoder | Registers an edge feature encoder in GraphGym. |
| register_stage | Registers a customized GNN stage in GraphGym. |
| register_head | Registers a GNN prediction head in GraphGym. |
| register_layer | Registers a GNN layer in GraphGym. |
| register_pooling | Registers a GNN global pooling/readout layer in GraphGym. |
| register_network | Registers a GNN model in GraphGym. |
| register_config | Registers a configuration group in GraphGym. |
| register_dataset | Registers a dataset in GraphGym. |
| register_loader | Registers a data loader in GraphGym. |
| register_optimizer | Registers an optimizer in GraphGym. |
| register_scheduler | Registers a learning rate scheduler in GraphGym. |
| register_loss | Registers a loss function in GraphGym. |
| register_train | Registers a training function in GraphGym. |
| register_metric | Register a metric function in GraphGym. |


**load_ckpt(*model: [Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)*, *optimizer: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Optimizer](https://docs.pytorch.org/docs/main/optim.html#torch.optim.Optimizer)] = None*, *scheduler: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*, *epoch: [int](https://docs.python.org/3/library/functions.html#int) = -1*) → [int](https://docs.python.org/3/library/functions.html#int)[[source]](../_modules/torch_geometric/graphgym/checkpoint.html#load_ckpt)**
: Loads the model checkpoint at a given epoch.


**Return type:**
: [int](https://docs.python.org/3/library/functions.html#int)


**save_ckpt(*model: [Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)*, *optimizer: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Optimizer](https://docs.pytorch.org/docs/main/optim.html#torch.optim.Optimizer)] = None*, *scheduler: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*, *epoch: [int](https://docs.python.org/3/library/functions.html#int) = 0*)[[source]](../_modules/torch_geometric/graphgym/checkpoint.html#save_ckpt)**
: Saves the model checkpoint at a given epoch.


**remove_ckpt(*epoch: [int](https://docs.python.org/3/library/functions.html#int) = -1*)[[source]](../_modules/torch_geometric/graphgym/checkpoint.html#remove_ckpt)**
: Removes the model checkpoint at a given epoch.


**clean_ckpt()[[source]](../_modules/torch_geometric/graphgym/checkpoint.html#clean_ckpt)**
: Removes all but the last model checkpoint.


**parse_args() → [Namespace](https://docs.python.org/3/library/argparse.html#argparse.Namespace)[[source]](../_modules/torch_geometric/graphgym/cmd_args.html#parse_args)**
: Parses the command line arguments.


**Return type:**
: [Namespace](https://docs.python.org/3/library/argparse.html#argparse.Namespace)


**set_cfg(*cfg*)[[source]](../_modules/torch_geometric/graphgym/config.html#set_cfg)**
: This function sets the default config value.


1. Note that for an experiment, only part of the arguments will be used
The remaining unused arguments won’t affect anything.
So feel free to register any argument in graphgym.contrib.config
2. We support *at most* two levels of configs, *e.g.*,
`cfg.dataset.name`.


**Returns:**
: Configuration use by the experiment.


**load_cfg(*cfg*, *args*)[[source]](../_modules/torch_geometric/graphgym/config.html#load_cfg)**
: Load configurations from file system and command line.


**Parameters:**
: - **cfg** (*CfgNode*) – Configuration node
- **args** (*ArgumentParser*) – Command argument parser


**dump_cfg(*cfg*)[[source]](../_modules/torch_geometric/graphgym/config.html#dump_cfg)**
: Dumps the config to the output directory specified in
`cfg.out_dir`.


**Parameters:**
: **cfg** (*CfgNode*) – Configuration node


**set_run_dir(*out_dir*)[[source]](../_modules/torch_geometric/graphgym/config.html#set_run_dir)**
: Create the directory for each random seed experiment run.


**Parameters:**
: **out_dir** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – Directory for output, specified in `cfg.out_dir`


**set_out_dir(*out_dir*, *fname*)[[source]](../_modules/torch_geometric/graphgym/config.html#set_out_dir)**
: Create the directory for full experiment run.


**Parameters:**
: - **out_dir** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – Directory for output, specified in `cfg.out_dir`
- **fname** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – Filename for the yaml format configuration file


**get_fname(*fname*)[[source]](../_modules/torch_geometric/graphgym/config.html#get_fname)**
: Extract filename from file name path.


**Parameters:**
: **fname** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – Filename for the yaml format configuration file


**init_weights(*m*)[[source]](../_modules/torch_geometric/graphgym/init.html#init_weights)**
: Performs weight initialization.


**Parameters:**
: **m** (*nn.Module*) – PyTorch module


**create_loader()[[source]](../_modules/torch_geometric/graphgym/loader.html#create_loader)**
: Create data loader object.


Returns: List of PyTorch data loaders


**set_printing()[[source]](../_modules/torch_geometric/graphgym/logger.html#set_printing)**
: Set up printing options.


**create_logger()[[source]](../_modules/torch_geometric/graphgym/logger.html#create_logger)**
: Create logger for the experiment.


**compute_loss(*pred*, *true*)[[source]](../_modules/torch_geometric/graphgym/loss.html#compute_loss)**
: Compute loss and prediction score.


**Parameters:**
: - **pred** (*torch.tensor*) – Unnormalized prediction
- **true** (*torch.tensor*) – Ground truth labels


Returns: Loss, normalized prediction score


**create_model(*to_device=True*, *dim_in=None*, *dim_out=None*) → GraphGymModule[[source]](../_modules/torch_geometric/graphgym/model_builder.html#create_model)**
: Create model for graph machine learning.


**Parameters:**
: - **to_device** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – Whether to transfer the model to the
specified device. (default: [True](https://docs.python.org/3/library/constants.html#True))
- **dim_in** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – Input dimension to the model
- **dim_out** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – Output dimension to the model

**Return type:**
: `GraphGymModule`


**create_optimizer(*params: [Iterator](https://docs.python.org/3/library/typing.html#typing.Iterator)[[Parameter](https://docs.pytorch.org/docs/main/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter)]*, *cfg: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)[[source]](../_modules/torch_geometric/graphgym/optim.html#create_optimizer)**
: Creates a config-driven optimizer.


**Return type:**
: [Any](https://docs.python.org/3/library/typing.html#typing.Any)


**create_scheduler(*optimizer: [Optimizer](https://docs.pytorch.org/docs/main/optim.html#torch.optim.Optimizer)*, *cfg: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)[[source]](../_modules/torch_geometric/graphgym/optim.html#create_scheduler)**
: Creates a config-driven learning rate scheduler.


**Return type:**
: [Any](https://docs.python.org/3/library/typing.html#typing.Any)


**train(*model: GraphGymModule*, *datamodule: GraphGymDataModule*, *logger: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, *trainer_config: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]] = None*)[[source]](../_modules/torch_geometric/graphgym/train.html#train)**
: Trains a GraphGym model using PyTorch Lightning.


**Parameters:**
: - **model** (*GraphGymModule*) – The GraphGym model.
- **datamodule** (*GraphGymDataModule*) – The GraphGym data module.
- **logger** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – Whether to enable logging during training.
(default: [True](https://docs.python.org/3/library/constants.html#True))
- **trainer_config** ([dict](https://docs.python.org/3/library/stdtypes.html#dict)*, **optional*) – Additional trainer configuration.


**register_base(*mapping: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/library/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]*, *key: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *module: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[None](https://docs.python.org/3/library/constants.html#None), [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)][[source]](../_modules/torch_geometric/graphgym/register.html#register_base)**
: Base function for registering a module in GraphGym.


**Parameters:**
: - **mapping** ([dict](https://docs.python.org/3/library/stdtypes.html#dict)) – Python dictionary to register the module.
hosting all the registered modules
- **key** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – The name of the module.
- **module** (*any**, **optional*) – The module. If set to [None](https://docs.python.org/3/library/constants.html#None), will return
a decorator to register a module.

**Return type:**
: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)]


**register_act(*key: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *module: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*)[[source]](../_modules/torch_geometric/graphgym/register.html#register_act)**
: Registers an activation function in GraphGym.


**register_node_encoder(*key: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *module: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*)[[source]](../_modules/torch_geometric/graphgym/register.html#register_node_encoder)**
: Registers a node feature encoder in GraphGym.


**register_edge_encoder(*key: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *module: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*)[[source]](../_modules/torch_geometric/graphgym/register.html#register_edge_encoder)**
: Registers an edge feature encoder in GraphGym.


**register_stage(*key: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *module: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*)[[source]](../_modules/torch_geometric/graphgym/register.html#register_stage)**
: Registers a customized GNN stage in GraphGym.


**register_head(*key: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *module: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*)[[source]](../_modules/torch_geometric/graphgym/register.html#register_head)**
: Registers a GNN prediction head in GraphGym.


**register_layer(*key: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *module: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*)[[source]](../_modules/torch_geometric/graphgym/register.html#register_layer)**
: Registers a GNN layer in GraphGym.


**register_pooling(*key: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *module: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*)[[source]](../_modules/torch_geometric/graphgym/register.html#register_pooling)**
: Registers a GNN global pooling/readout layer in GraphGym.


**register_network(*key: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *module: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*)[[source]](../_modules/torch_geometric/graphgym/register.html#register_network)**
: Registers a GNN model in GraphGym.


**register_config(*key: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *module: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*)[[source]](../_modules/torch_geometric/graphgym/register.html#register_config)**
: Registers a configuration group in GraphGym.


**register_dataset(*key: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *module: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*)[[source]](../_modules/torch_geometric/graphgym/register.html#register_dataset)**
: Registers a dataset in GraphGym.


**register_loader(*key: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *module: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*)[[source]](../_modules/torch_geometric/graphgym/register.html#register_loader)**
: Registers a data loader in GraphGym.


**register_optimizer(*key: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *module: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*)[[source]](../_modules/torch_geometric/graphgym/register.html#register_optimizer)**
: Registers an optimizer in GraphGym.


**register_scheduler(*key: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *module: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*)[[source]](../_modules/torch_geometric/graphgym/register.html#register_scheduler)**
: Registers a learning rate scheduler in GraphGym.


**register_loss(*key: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *module: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*)[[source]](../_modules/torch_geometric/graphgym/register.html#register_loss)**
: Registers a loss function in GraphGym.


**register_train(*key: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *module: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*)[[source]](../_modules/torch_geometric/graphgym/register.html#register_train)**
: Registers a training function in GraphGym.


**register_metric(*key: [str](https://docs.python.org/3/library/stdtypes.html#str)*, *module: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] = None*)[[source]](../_modules/torch_geometric/graphgym/register.html#register_metric)**
: Register a metric function in GraphGym.


## Model Modules


| IntegerFeatureEncoder | Provides an encoder for integer node features. |
| --- | --- |
| AtomEncoder | The atom encoder used in OGB molecule dataset. |
| BondEncoder | The bond encoder used in OGB molecule dataset. |
| GNNLayer | Creates a GNN layer, given the specified input and output dimensions and the underlying configuration incfg. |
| GNNPreMP | Creates a NN layer used before message passing, given the specified input and output dimensions and the underlying configuration incfg. |
| GNNStackStage | Stacks a number of GNN layers. |
| FeatureEncoder | Encodes node and edge features, given the specified input dimension and the underlying configuration incfg. |
| GNN | A general Graph Neural Network (GNN) model. |
| GNNNodeHead | A GNN prediction head for node-level prediction tasks. |
| GNNEdgeHead | A GNN prediction head for edge-level/link-level prediction tasks. |
| GNNGraphHead | A GNN prediction head for graph-level prediction tasks. |
| GeneralLayer | A general wrapper for layers. |
| GeneralMultiLayer | A general wrapper class for a stacking multiple NN layers. |
| Linear | A basic Linear layer. |
| BatchNorm1dNode | A batch normalization layer for node-level features. |
| BatchNorm1dEdge | A batch normalization layer for edge-level features. |
| MLP | A basic MLP model. |
| GCNConv | A Graph Convolutional Network (GCN) layer. |
| SAGEConv | A GraphSAGE layer. |
| GATConv | A Graph Attention Network (GAT) layer. |
| GINConv | A Graph Isomorphism Network (GIN) layer. |
| SplineConv | A SplineCNN layer. |
| GeneralConv | A general GNN layer. |
| GeneralEdgeConv | A general GNN layer with edge feature support. |
| GeneralSampleEdgeConv | A general GNN layer that supports edge features and edge sampling. |
| global_add_pool | Returns batch-wise graph-level-outputs by adding node features across the node dimension. |
| global_mean_pool | Returns batch-wise graph-level-outputs by averaging node features across the node dimension. |
| global_max_pool | Returns batch-wise graph-level-outputs by taking the channel-wise maximum across the node dimension. |


***class *IntegerFeatureEncoder(*emb_dim: [int](https://docs.python.org/3/library/functions.html#int)*, *num_classes: [int](https://docs.python.org/3/library/functions.html#int)*)[[source]](../_modules/torch_geometric/graphgym/models/encoder.html#IntegerFeatureEncoder)**
: Provides an encoder for integer node features.


**Parameters:**
: - **emb_dim** ([int](https://docs.python.org/3/library/functions.html#int)) – The output embedding dimension.
- **num_classes** ([int](https://docs.python.org/3/library/functions.html#int)) – The number of classes/integers.


Example


```
>>> encoder = IntegerFeatureEncoder(emb_dim=16, num_classes=10)
>>> batch = torch.randint(0, 10, (10, 2))
>>> encoder(batch).size()
torch.Size([10, 16])
```


***class *AtomEncoder(*emb_dim*, **args*, ***kwargs*)[[source]](../_modules/torch_geometric/graphgym/models/encoder.html#AtomEncoder)**
: The atom encoder used in OGB molecule dataset.


**Parameters:**
: **emb_dim** ([int](https://docs.python.org/3/library/functions.html#int)) – The output embedding dimension.


Example


```
>>> encoder = AtomEncoder(emb_dim=16)
>>> batch = torch.randint(0, 10, (10, 3))
>>> encoder(batch).size()
torch.Size([10, 16])
```


***class *BondEncoder(*emb_dim: [int](https://docs.python.org/3/library/functions.html#int)*)[[source]](../_modules/torch_geometric/graphgym/models/encoder.html#BondEncoder)**
: The bond encoder used in OGB molecule dataset.


**Parameters:**
: **emb_dim** ([int](https://docs.python.org/3/library/functions.html#int)) – The output embedding dimension.


Example


```
>>> encoder = BondEncoder(emb_dim=16)
>>> batch = torch.randint(0, 10, (10, 3))
>>> encoder(batch).size()
torch.Size([10, 16])
```


**GNNLayer(*dim_in: [int](https://docs.python.org/3/library/functions.html#int)*, *dim_out: [int](https://docs.python.org/3/library/functions.html#int)*, *has_act: [bool](https://docs.python.org/3/library/functions.html#bool) = True*) → GeneralLayer[[source]](../_modules/torch_geometric/graphgym/models/gnn.html#GNNLayer)**
: Creates a GNN layer, given the specified input and output dimensions
and the underlying configuration in `cfg`.


**Parameters:**
: - **dim_in** ([int](https://docs.python.org/3/library/functions.html#int)) – The input dimension
- **dim_out** ([int](https://docs.python.org/3/library/functions.html#int)) – The output dimension.
- **has_act** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – Whether to apply an activation function
after the layer. (default: [True](https://docs.python.org/3/library/constants.html#True))

**Return type:**
: GeneralLayer


**GNNPreMP(*dim_in: [int](https://docs.python.org/3/library/functions.html#int)*, *dim_out: [int](https://docs.python.org/3/library/functions.html#int)*, *num_layers: [int](https://docs.python.org/3/library/functions.html#int)*) → GeneralMultiLayer[[source]](../_modules/torch_geometric/graphgym/models/gnn.html#GNNPreMP)**
: Creates a NN layer used before message passing, given the specified
input and output dimensions and the underlying configuration in `cfg`.


**Parameters:**
: - **dim_in** ([int](https://docs.python.org/3/library/functions.html#int)) – The input dimension
- **dim_out** ([int](https://docs.python.org/3/library/functions.html#int)) – The output dimension.
- **num_layers** ([int](https://docs.python.org/3/library/functions.html#int)) – The number of layers.

**Return type:**
: GeneralMultiLayer


***class *GNNStackStage(*dim_in*, *dim_out*, *num_layers*)[[source]](../_modules/torch_geometric/graphgym/models/gnn.html#GNNStackStage)**
: Stacks a number of GNN layers.


**Parameters:**
: - **dim_in** ([int](https://docs.python.org/3/library/functions.html#int)) – The input dimension
- **dim_out** ([int](https://docs.python.org/3/library/functions.html#int)) – The output dimension.
- **num_layers** ([int](https://docs.python.org/3/library/functions.html#int)) – The number of layers.


***class *FeatureEncoder(*dim_in: [int](https://docs.python.org/3/library/functions.html#int)*)[[source]](../_modules/torch_geometric/graphgym/models/gnn.html#FeatureEncoder)**
: Encodes node and edge features, given the specified input dimension and
the underlying configuration in `cfg`.


**Parameters:**
: **dim_in** ([int](https://docs.python.org/3/library/functions.html#int)) – The input feature dimension.


***class *GNN(*dim_in: [int](https://docs.python.org/3/library/functions.html#int)*, *dim_out: [int](https://docs.python.org/3/library/functions.html#int)*, ***kwargs*)[[source]](../_modules/torch_geometric/graphgym/models/gnn.html#GNN)**
: A general Graph Neural Network (GNN) model.


The GNN model consists of three main components:


1. An encoder to transform input features into a fixed-size embedding
space.
2. A processing or message passing stage for information exchange between
nodes.
3. A head to produce the final output features/predictions.


The configuration of each component is determined by the underlying
configuration in `cfg`.


**Parameters:**
: - **dim_in** ([int](https://docs.python.org/3/library/functions.html#int)) – The input feature dimension.
- **dim_out** ([int](https://docs.python.org/3/library/functions.html#int)) – The output feature dimension.
- ****kwargs** (*optional*) – Additional keyword arguments.


***class *GNNNodeHead(*dim_in: [int](https://docs.python.org/3/library/functions.html#int)*, *dim_out: [int](https://docs.python.org/3/library/functions.html#int)*)[[source]](../_modules/torch_geometric/graphgym/models/head.html#GNNNodeHead)**
: A GNN prediction head for node-level prediction tasks.


**Parameters:**
: - **dim_in** ([int](https://docs.python.org/3/library/functions.html#int)) – The input feature dimension.
- **dim_out** ([int](https://docs.python.org/3/library/functions.html#int)) – The output feature dimension.


***class *GNNEdgeHead(*dim_in: [int](https://docs.python.org/3/library/functions.html#int)*, *dim_out: [int](https://docs.python.org/3/library/functions.html#int)*)[[source]](../_modules/torch_geometric/graphgym/models/head.html#GNNEdgeHead)**
: A GNN prediction head for edge-level/link-level prediction tasks.


**Parameters:**
: - **dim_in** ([int](https://docs.python.org/3/library/functions.html#int)) – The input feature dimension.
- **dim_out** ([int](https://docs.python.org/3/library/functions.html#int)) – The output feature dimension.


***class *GNNGraphHead(*dim_in: [int](https://docs.python.org/3/library/functions.html#int)*, *dim_out: [int](https://docs.python.org/3/library/functions.html#int)*)[[source]](../_modules/torch_geometric/graphgym/models/head.html#GNNGraphHead)**
: A GNN prediction head for graph-level prediction tasks.
A post message passing layer (as specified by `cfg.gnn.post_mp`) is
used to transform the pooled graph-level embeddings using an MLP.


**Parameters:**
: - **dim_in** ([int](https://docs.python.org/3/library/functions.html#int)) – The input feature dimension.
- **dim_out** ([int](https://docs.python.org/3/library/functions.html#int)) – The output feature dimension.


***class *GeneralLayer(*name*, *layer_config: LayerConfig*, ***kwargs*)[[source]](../_modules/torch_geometric/graphgym/models/layer.html#GeneralLayer)**
: A general wrapper for layers.


**Parameters:**
: - **name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – The registered name of the layer.
- **layer_config** (*LayerConfig*) – The configuration of the layer.
- ****kwargs** (*optional*) – Additional keyword arguments.


***class *GeneralMultiLayer(*name*, *layer_config: LayerConfig*, ***kwargs*)[[source]](../_modules/torch_geometric/graphgym/models/layer.html#GeneralMultiLayer)**
: A general wrapper class for a stacking multiple NN layers.


**Parameters:**
: - **name** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – The registered name of the layer.
- **layer_config** (*LayerConfig*) – The configuration of the layer.
- ****kwargs** (*optional*) – Additional keyword arguments.


***class *Linear(*layer_config: LayerConfig*, ***kwargs*)[[source]](../_modules/torch_geometric/graphgym/models/layer.html#Linear)**
: A basic Linear layer.


**Parameters:**
: - **layer_config** (*LayerConfig*) – The configuration of the layer.
- ****kwargs** (*optional*) – Additional keyword arguments.


***class *BatchNorm1dNode(*layer_config: LayerConfig*)[[source]](../_modules/torch_geometric/graphgym/models/layer.html#BatchNorm1dNode)**
: A batch normalization layer for node-level features.


**Parameters:**
: **layer_config** (*LayerConfig*) – The configuration of the layer.


***class *BatchNorm1dEdge(*layer_config: LayerConfig*)[[source]](../_modules/torch_geometric/graphgym/models/layer.html#BatchNorm1dEdge)**
: A batch normalization layer for edge-level features.


**Parameters:**
: **layer_config** (*LayerConfig*) – The configuration of the layer.


***class *MLP(*layer_config: LayerConfig*, ***kwargs*)[[source]](../_modules/torch_geometric/graphgym/models/layer.html#MLP)**
: A basic MLP model.


**Parameters:**
: - **layer_config** (*LayerConfig*) – The configuration of the layer.
- ****kwargs** (*optional*) – Additional keyword arguments.


***class *GCNConv(*layer_config: LayerConfig*, ***kwargs*)[[source]](../_modules/torch_geometric/graphgym/models/layer.html#GCNConv)**
: A Graph Convolutional Network (GCN) layer.


***class *SAGEConv(*layer_config: LayerConfig*, ***kwargs*)[[source]](../_modules/torch_geometric/graphgym/models/layer.html#SAGEConv)**
: A GraphSAGE layer.


***class *GATConv(*layer_config: LayerConfig*, ***kwargs*)[[source]](../_modules/torch_geometric/graphgym/models/layer.html#GATConv)**
: A Graph Attention Network (GAT) layer.


***class *GINConv(*layer_config: LayerConfig*, ***kwargs*)[[source]](../_modules/torch_geometric/graphgym/models/layer.html#GINConv)**
: A Graph Isomorphism Network (GIN) layer.


***class *SplineConv(*layer_config: LayerConfig*, ***kwargs*)[[source]](../_modules/torch_geometric/graphgym/models/layer.html#SplineConv)**
: A SplineCNN layer.


***class *GeneralConv(*layer_config: LayerConfig*, ***kwargs*)[[source]](../_modules/torch_geometric/graphgym/models/layer.html#GeneralConv)**
: A general GNN layer.


***class *GeneralEdgeConv(*layer_config: LayerConfig*, ***kwargs*)[[source]](../_modules/torch_geometric/graphgym/models/layer.html#GeneralEdgeConv)**
: A general GNN layer with edge feature support.


***class *GeneralSampleEdgeConv(*layer_config: LayerConfig*, ***kwargs*)[[source]](../_modules/torch_geometric/graphgym/models/layer.html#GeneralSampleEdgeConv)**
: A general GNN layer that supports edge features and edge sampling.


**global_add_pool(*x: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *batch: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *size: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/nn/pool/glob.html#global_add_pool)**
: Returns batch-wise graph-level-outputs by adding node features
across the node dimension.


For a single graph $\mathcal{G}_i$, its output is computed by


$$
\mathbf{r}_i = \sum_{n=1}^{N_i} \mathbf{x}_n.
$$


Functional method of the
[SumAggregation](../generated/torch_geometric.nn.aggr.SumAggregation.html#torch_geometric.nn.aggr.SumAggregation) module.


**Parameters:**
: - **x** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – Node feature matrix
$\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}$.
- **batch** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The batch vector
$\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N$, which assigns
each node to a specific example.
- **size** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of examples $B$.
Automatically calculated if not given. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


**global_mean_pool(*x: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *batch: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *size: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/nn/pool/glob.html#global_mean_pool)**
: Returns batch-wise graph-level-outputs by averaging node features
across the node dimension.


For a single graph $\mathcal{G}_i$, its output is computed by


$$
\mathbf{r}_i = \frac{1}{N_i} \sum_{n=1}^{N_i} \mathbf{x}_n.
$$


Functional method of the
[MeanAggregation](../generated/torch_geometric.nn.aggr.MeanAggregation.html#torch_geometric.nn.aggr.MeanAggregation) module.


**Parameters:**
: - **x** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – Node feature matrix
$\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}$.
- **batch** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The batch vector
$\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N$, which assigns
each node to a specific example.
- **size** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of examples $B$.
Automatically calculated if not given. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


**global_max_pool(*x: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, *batch: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)]*, *size: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*) → [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)[[source]](../_modules/torch_geometric/nn/pool/glob.html#global_max_pool)**
: Returns batch-wise graph-level-outputs by taking the channel-wise
maximum across the node dimension.


For a single graph $\mathcal{G}_i$, its output is computed by


$$
\mathbf{r}_i = \mathrm{max}_{n=1}^{N_i} \, \mathbf{x}_n.
$$


Functional method of the
[MaxAggregation](../generated/torch_geometric.nn.aggr.MaxAggregation.html#torch_geometric.nn.aggr.MaxAggregation) module.


**Parameters:**
: - **x** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)) – Node feature matrix
$\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}$.
- **batch** ([torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)*, **optional*) – The batch vector
$\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N$, which assigns
each element to a specific example.
- **size** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of examples $B$.
Automatically calculated if not given. (default: [None](https://docs.python.org/3/library/constants.html#None))

**Return type:**
: [Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor)


## Utility Modules


| agg_runs | Aggregate over different random seeds of a single experiment. |
| --- | --- |
| agg_batch | Aggregate across results from multiple experiments via grid search. |
| params_count | Computes the number of parameters. |
| match_baseline_cfg | Match the computational budget of a given baseline model. |
| get_current_gpu_usage | Get the current GPU memory usage. |
| auto_select_device | Auto select device for the current experiment. |
| is_eval_epoch | Determines if the model should be evaluated at the current epoch. |
| is_ckpt_epoch | Determines if the model should be evaluated at the current epoch. |
| dict_to_json | Dump aPythondictionary to a JSON file. |
| dict_list_to_json | Dump a list ofPythondictionaries to a JSON file. |
| dict_to_tb | Add a dictionary of statistics to a Tensorboard writer. |
| makedirs_rm_exist | Make a directory, remove any existing data. |
| dummy_context | Default context manager that does nothing. |


**agg_runs(*dir*, *metric_best='auto'*)[[source]](../_modules/torch_geometric/graphgym/utils/agg_runs.html#agg_runs)**
: Aggregate over different random seeds of a single experiment.


**Parameters:**
: - **dir** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – Directory of the results, containing 1 experiment
- **metric_best** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The metric for selecting the best
- **Options** (*validation performance.*) – auto, accuracy, auc.


**agg_batch(*dir*, *metric_best='auto'*)[[source]](../_modules/torch_geometric/graphgym/utils/agg_runs.html#agg_batch)**
: Aggregate across results from multiple experiments via grid search.


**Parameters:**
: - **dir** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – Directory of the results, containing multiple experiments
- **metric_best** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – The metric for selecting the best
- **Options** (*validation performance.*) – auto, accuracy, auc.


**params_count(*model*)[[source]](../_modules/torch_geometric/graphgym/utils/comp_budget.html#params_count)**
: Computes the number of parameters.


**Parameters:**
: **model** (*nn.Module*) – PyTorch model


**match_baseline_cfg(*cfg_dict*, *cfg_dict_baseline*, *verbose=True*)[[source]](../_modules/torch_geometric/graphgym/utils/comp_budget.html#match_baseline_cfg)**
: Match the computational budget of a given baseline model. The current
configuration dictionary will be modified and returned.


**Parameters:**
: - **cfg_dict** ([dict](https://docs.python.org/3/library/stdtypes.html#dict)) – Current experiment’s configuration
- **cfg_dict_baseline** ([dict](https://docs.python.org/3/library/stdtypes.html#dict)) – Baseline configuration
- **verbose** ([str](https://docs.python.org/3/library/stdtypes.html#str)*, **optional*) – If printing matched parameter conunts


**get_current_gpu_usage()[[source]](../_modules/torch_geometric/graphgym/utils/device.html#get_current_gpu_usage)**
: Get the current GPU memory usage.


**auto_select_device()[[source]](../_modules/torch_geometric/graphgym/utils/device.html#auto_select_device)**
: Auto select device for the current experiment.


**is_eval_epoch(*cur_epoch*)[[source]](../_modules/torch_geometric/graphgym/utils/epoch.html#is_eval_epoch)**
: Determines if the model should be evaluated at the current epoch.


**is_ckpt_epoch(*cur_epoch*)[[source]](../_modules/torch_geometric/graphgym/utils/epoch.html#is_ckpt_epoch)**
: Determines if the model should be evaluated at the current epoch.


**dict_to_json(*dict*, *fname*)[[source]](../_modules/torch_geometric/graphgym/utils/io.html#dict_to_json)**
: Dump a Python dictionary to a JSON file.


**Parameters:**
: - **dict** ([dict](https://docs.python.org/3/library/stdtypes.html#dict)) – The Python dictionary.
- **fname** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – The output file name.


**dict_list_to_json(*dict_list*, *fname*)[[source]](../_modules/torch_geometric/graphgym/utils/io.html#dict_list_to_json)**
: Dump a list of Python dictionaries to a JSON file.


**Parameters:**
: - **dict_list** (*list of dict*) – List of Python dictionaries.
- **fname** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – the output file name.


**dict_to_tb(*dict*, *writer*, *epoch*)[[source]](../_modules/torch_geometric/graphgym/utils/io.html#dict_to_tb)**
: Add a dictionary of statistics to a Tensorboard writer.


**Parameters:**
: - **dict** ([dict](https://docs.python.org/3/library/stdtypes.html#dict)) – Statistics of experiments, the keys are attribute names,
- **values** (*the values are the attribute*) –
- **writer** – Tensorboard writer object
- **epoch** ([int](https://docs.python.org/3/library/functions.html#int)) – The current epoch


**makedirs_rm_exist(*dir*)[[source]](../_modules/torch_geometric/graphgym/utils/io.html#makedirs_rm_exist)**
: Make a directory, remove any existing data.


**Parameters:**
: **dir** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – The directory to be created.


***class *dummy_context[[source]](../_modules/torch_geometric/graphgym/utils/tools.html#dummy_context)**
: Default context manager that does nothing.


