TorchScript is a way to create serializable and optimizable models from PyTorch code.
Any TorchScript program can be saved from a Python process and loaded in a process where there is no Python dependency.
If you are unfamilar with TorchScript, we recommend to read the official “[Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)” tutorial first.


## Converting GNN Models


> **Note:** From PyG 2.5 (and onwards), GNN layers are now fully compatible with `torch.jit.script()` without any modification needed.
If you are on an earlier version of PyG, consider to convert your GNN layers into “jittable” instances first by calling [jittable()](../generated/torch_geometric.nn.conv.MessagePassing.html#torch_geometric.nn.conv.MessagePassing.jittable).


Converting your PyG model to a TorchScript program is straightforward and requires only a few code changes.
Let’s consider the following model:


```
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GNN(dataset.num_features, dataset.num_classes)
```


The instantiated model can now be directly passed into `torch.jit.script()`:


```
model = torch.jit.script(model)
```


That is all you need to know on how to convert your PyG models to TorchScript programs.
You can have a further look at our JIT examples that show-case how to obtain TorchScript programs for [node](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/jit/gat.py) and [graph classification](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/jit/gin.py) models.


## Creating Jittable GNN Operators


All PyG [MessagePassing](../generated/torch_geometric.nn.conv.MessagePassing.html#torch_geometric.nn.conv.MessagePassing) operators are tested to be convertible to a TorchScript program.
However, if you want your own GNN module to be compatible with `torch.jit.script()`, you need to account for the following two things:


1. As one would expect, your `forward()` code may need to be adjusted so that it passes the TorchScript compiler requirements, *e.g.*, by adding type notations.
2. You need to tell the [MessagePassing](../generated/torch_geometric.nn.conv.MessagePassing.html#torch_geometric.nn.conv.MessagePassing) module the types that you pass to its [propagate()](../generated/torch_geometric.nn.conv.MessagePassing.html#torch_geometric.nn.conv.MessagePassing.propagate) function.
This can be achieved in two different ways:


1. Declaring the type of propagation arguments in a dictionary called `propagate_type`:


> ```
> from typing import Optional
> from torch import Tensor
> from torch_geometric.nn import MessagePassing
>
> class MyConv(MessagePassing):
>     propagate_type = {'x': Tensor, 'edge_weight': Optional[Tensor] }
>
>     def forward(
>         self,
>         x: Tensor,
>         edge_index: Tensor,
>         edge_weight: Optional[Tensor] = None,
>     ) -> Tensor:
>         return self.propagate(edge_index, x=x, edge_weight=edge_weight)
> ```


1. Declaring the type of propagation arguments as a comment inside your module:


> ```
> from typing import Optional
> from torch import Tensor
> from torch_geometric.nn import MessagePassing
>
> class MyConv(MessagePassing):
>     def forward(
>         self,
>         x: Tensor,
>         edge_index: Tensor,
>         edge_weight: Optional[Tensor] = None,
>     ) -> Tensor:
>         # propagate_type: (x: Tensor, edge_weight: Optional[Tensor])
>         return self.propagate(edge_index, x=x, edge_weight=edge_weight)
> ```


If none of these options are given, the [MessagePassing](../generated/torch_geometric.nn.conv.MessagePassing.html#torch_geometric.nn.conv.MessagePassing) module will infer the arguments of [propagate()](../generated/torch_geometric.nn.conv.MessagePassing.html#torch_geometric.nn.conv.MessagePassing.propagate) to be of type [torch.Tensor](https://docs.pytorch.org/docs/main/tensors.html#torch.Tensor) (mimicking the default type that TorchScript is inferring for non-annotated arguments).


