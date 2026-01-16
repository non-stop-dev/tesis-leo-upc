GraphGym is a platform for **designing and evaluating Graph Neural Networks (GNNs)**, as originally proposed in the [“Design Space for Graph Neural Networks”](https://arxiv.org/abs/2011.08843) paper.
We now officially support GraphGym as part of of PyG.


> **Warning:** GraphGym API may change in the future as we are continuously working on better and deeper integration with PyG.


## Highlights


1. **Highly modularized pipeline for GNN:**


- **Data:** Data loading and data splitting
- **Model:** Modularized GNN implementations
- **Tasks:** Node-level, edge-level and graph-level tasks
- **Evaluation:** Accuracy, ROC AUC, …
2. **Reproducible experiment configuration:**


- Each experiment is *fully described by a configuration file*
3. **Scalable experiment management:**


- Easily launch *thousands of GNN experiments in parallel*
- *Auto-generate* experiment analyses and figures across random seeds and experiments
4. **Flexible user customization:**


- Easily *register your own modules*, such as data loaders, GNN layers, loss functions, etc


## Why GraphGym?


**TL;DR:** GraphGym is great for GNN beginners, domain experts and GNN researchers.


**Scenario 1:** You are a beginner to graph representation learning and want to understand how GNNs work:


You probably have read many exciting papers on GNNs, and try to write your own GNN implementation.
Even if using raw PyG, you still have to code up the essential pipeline on your own.
GraphGym is a perfect place for your to start learning about *standardized GNN implementation and evaluation*.


[](../_images/graphgym_design_space.png)

***Figure 1:** Modularized GNN implementation.*


**Scenario 2:** You want to apply GNNs to your exciting application:


You probably know that there are hundreds of possible GNN models, and selecting the best model is notoriously hard.
Even worse, the [GraphGym paper](https://arxiv.org/abs/2011.08843) shows that the best GNN designs for different tasks differ drastically.
GraphGym provides a *simple interface to try out thousands of GNNs in parallel* and understand the best designs for your specific task.
GraphGym also recommends a “go-to” GNN design space, after investigating 10 million GNN model-task combinations.


[](../_images/graphgym_results.png)

***Figure 2:** A guideline for desirable GNN design choices.*


**Scenario 3:** You are a GNN researcher, who wants to innovate new GNN models or propose new GNN tasks:


Say you have proposed a new GNN layer `ExampleConv`.
GraphGym can help you convincingly argue that `ExampleConv` is better than, *e.g.*, [GCNConv](../generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv):
When randomly sampling from 10 million possible model-task combinations, how often will `ExampleConv` will outperform [GCNConv](../generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv) when everything else is fixed (including computational costs)?
Moreover, GraphGym can help you easily do hyper-parameter search, and *visualize* what design choices are better.
In sum, GraphGym can greatly facilitate your GNN research.


[](../_images/graphgym_evaluation.png)

***Figure 3:** Evaluation of a given GNN design dimension, *e.g.*, `BatchNorm`.*


## Basic Usage


> **Note:** For using GraphGym, PyG requires additional dependencies.
You can install those by running `pip install torch-geometric[graphgym]`.


To use GraphGym, you need to clone PyG from GitHub, then change to the `graphgym/` directory.


```
git clone https://github.com/pyg-team/pytorch_geometric.git
cd pytorch_geometric/graphgym
```


1. **Run a single experiment:**
Run an experiment using GraphGym via `run_single.sh`.
Configurations are specified in `configs/pyg/example_node.yaml`.
The default experiment is about node classification on the [Planetoid](../generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid) datasets (using a random 80/20 train/validation split).


```
bash run_single.sh # run a single experiment
```
2. **Run a batch of experiments:**
Run a batch of experiments using GraphGym via `run_batch.sh`.
Configurations are specified in `configs/pyg/example_node.yaml` (controls the basic architecture) and `grids/example.txt` (controls how to do grid search).
The experiment examines 96 models in the recommended GNN design space, on 2 graph classification datasets.
Each experiment is repeated 3 times, and we set up that 8 jobs can be concurrently run.
Depending on your infrastructure, finishing all the experiments may take a long time;
you can quit the experiment via `Ctrl-C` (GraphGym will properly kill all the processes).


```
bash run_batch.sh # run a batch of experiments
```
3. **Run GraphGym with CPU backend:**
GraphGym supports CPU backend as well – you only need to add the line `accelerator: cpu` to the `*.yaml` file.


## In-Depth Usage


To use GraphGym, you need to clone PyG from GitHub, then change to the `graphgym/` directory.


```
git clone https://github.com/pyg-team/pytorch_geometric.git
cd pytorch_geometric/graphgym
```


1. **Run a single experiment:**
A full example is specified in `run_single.sh`.


1. **Specify a configuration file:**
In GraphGym, an experiment is fully specified by a `*.yaml` file.
Unspecified configurations in the `*.yaml` file will be populated by the default values in [torch_geometric.graphgym.set_cfg()](../modules/graphgym.html#torch_geometric.graphgym.set_cfg).
For example, in `configs/pyg/example_node.yaml`, there are configurations for the dataset, training procedure, model, etc.
Concrete description for each configuration is described in [set_cfg()](../modules/graphgym.html#torch_geometric.graphgym.set_cfg).
2. **Launch an experiment:**
For example, in `run_single.sh`:


```
python main.py --cfg configs/pyg/example_node.yaml --repeat 3
```


You can specify the number of different random seeds to repeat via `--repeat`.
3. **Understand the results:**
Experimental results will be automatically saved in `results/${CONFIG_NAME}/`.
In the example above, this amounts to `results/pyg/example_node/`.
Results for different random seeds will be saved in different subdirectories, *e.g.*, `results/pyg/example_node/2`.
The aggregated results over all the random seeds are *automatically* generated into `results/example/agg`, including the mean and standard deviation `_std` for each metric.
Train/validation/test results are further saved into subdirectories, such as `results/example/agg/val`.
Here, `stats.json` stores the results after each epoch aggregated across random seeds, and `best.json` stores the results of *the epoch with the highest validation accuracy*.
2. **Run a batch of experiments:**
A full example is specified in `run_batch.sh`.


1. **Specify a base file:**
GraphGym supports running a batch of experiments.
To start, a user needs to select a base architecture via `--config`.
The batch of experiments will be created by perturbing certain configurations of the base architecture.
2. **(Optionally) specify a base file for computational budget:**
Additionally, GraphGym allows a user to select a base architecture to *control the computational budget* for the grid search via `--config_budget`.
The computational budget is currently measured by the number of trainable parameters, and the control is achieved by auto-adjusting the hidden dimensionality of the underlying GNN.
If no `--config_budget` is provided, GraphGym will not control the computational budget.
3. **Specify a grid file:**
A grid file describes how to perturb the base file, in order to generate the batch of experiments.
For example, the base file could specify an experiment of a 3-layer GCN for node classification on the [Planetoid](../generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid) datasets.
Then, the grid file specifies how to perturb the experiment along different dimensions, such as the number of layers, the model architecture, the dataset, the level of task, etc.
4. **Generate configuration files for the batch of experiments** based on the information specified above:
For example, in `run_batch.sh`:


```
python configs_gen.py --config configs/${DIR}/${CONFIG}.yaml \
  --config_budget configs/${DIR}/${CONFIG}.yaml \
  --grid grids/${DIR}/${GRID}.txt \
  --out_dir configs
```
5. **Launch the batch of experiments:**
For example, in `run_batch.sh`:


```
bash parallel.sh configs/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP
```


Each experiment will be repeated for `$REPEAT` times.
We implemented a queue system to sequentially launch all the jobs, with `$MAX_JOBS` concurrent jobs running at the same time.
In practice, our system works great when handling thousands of jobs.
6. **Understand the results:**
Experimental results will be automatically saved in directory `results/${CONFIG_NAME}_grid_${GRID_NAME}/`.
In the example above, this amounts to `results/pyg/example_grid_example/`.
After running each experiment, GraphGym additionally automatically averages across different models, saved in `results/pyg/example_grid_example/agg`.
There, `val.csv` represents the validation accuracy for each model configuration at the *final* epoch,
`val_best.csv` represents the results at the epoch with the highest average validation accuracy, and
`val_best_epoch.csv` represents the results at the epoch with the highest validation accuracy averaged over different random seeds.
When a test set split is provided, `test.csv` represents the test accuracy for each model configuration at the *final* epoch,
`test_best.csv` represents the test set results at the epoch with the highest average validation accuracy, and
`test_best_epoch.csv` represents the test set results at the epoch with the highest validation accuracy averaged over different random seeds.


## Customizing GraphGym


A highlight of GraphGym is that it allows you to easily register customized modules.
For each project, you can have a unique GraphGym copy with different customized modules.
For example, the [“Design Space for Graph Neural Networks”](https://arxiv.org/abs/2011.08843) and [“Identity-aware Graph Neural Networks”](https://arxiv.org/abs/2101.10320) papers represent two successful projects using customized GraphGym, and you may find more details about them [here](https://github.com/snap-stanford/GraphGym#use-case-design-space-for-graph-neural-networks-neurips-2020-spotlight).
Eventually, every GraphGym-powered project will be unique.


There are two ways for customizing GraphGym:


1. Use the `graphgym/custom_graphgym` directory outside the PyG package:
You can register your customized modules here without touching PyG. This use case will be great for your own customized project.
2. Use the `torch_geometric/graphgym/contrib` directory inside the PyG package:
If you have come up with a nice customized module, you can directly copy your files into `torch_geometric/graphgym/contrib`, and **create a pull request** to PyG.
This way, your idea can ship with PyG installations, and will have a much higher visibility and impact.


Concretely, the supported customized modules includes


- Activations: `custom_graphgym/act/`
- Customized configurations: `custom_graphgym/config/`
- Feature augmentations: `custom_graphgym/feature_augment/`
- Feature encoders: `custom_graphgym/feature_encoder/`
- GNN heads: `custom_graphgym/head/`
- GNN layers: `custom_graphgym/layer/`
- Data loaders: `custom_graphgym/loader/`
- Loss functions: `custom_graphgym/loss/`
- GNN network architectures: `custom_graphgym/network/`
- Optimizers: `custom_graphgym/optimizer/`
- GNN global pooling layers (for graph classification only): `custom_graphgym/pooling/`
- GNN stages: `custom_graphgym/stage/`
- GNN training pipelines: `custom_graphgym/train/`
- Data transformations: `custom_graphgym/transform/`


Within each directory, at least one example is provided that shows how to register customized modules via `torch_geometric.graphgym.register()`.
Note that new customized modules may result in new configurations.
In these cases, new configuration fields can be registered via `custom_graphgym/config/`.


