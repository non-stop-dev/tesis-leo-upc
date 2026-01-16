## Tensor Objects


| Index | A one-dimensionalindextensor with additional (meta)data attached. |
| --- | --- |
| EdgeIndex | A COOedge_indextensor with additional (meta)data attached. |
| HashTensor | Atorch.Tensorthat can be referenced by arbitrary keys rather than indices in the first dimension. |


## Functions


**seed_everything(*seed: [int](https://docs.python.org/3/library/functions.html#int)*) → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torch_geometric/seed.html#seed_everything)**
: Sets the seed for generating random numbers in PyTorch,
`numpy` and Python.


**Parameters:**
: **seed** ([int](https://docs.python.org/3/library/functions.html#int)) – The desired seed.

**Return type:**
: [None](https://docs.python.org/3/library/constants.html#None)


**get_home_dir() → [str](https://docs.python.org/3/library/stdtypes.html#str)[[source]](../_modules/torch_geometric/home.html#get_home_dir)**
: Get the cache directory used for storing all PyG-related data.


If set_home_dir() is not called, the path is given by the environment
variable `$PYG_HOME` which defaults to `"~/.cache/pyg"`.


**Return type:**
: [str](https://docs.python.org/3/library/stdtypes.html#str)


**set_home_dir(*path: [str](https://docs.python.org/3/library/stdtypes.html#str)*) → [None](https://docs.python.org/3/library/constants.html#None)[[source]](../_modules/torch_geometric/home.html#set_home_dir)**
: Set the cache directory used for storing all PyG-related data.


**Parameters:**
: **path** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – The path to a local folder.

**Return type:**
: [None](https://docs.python.org/3/library/constants.html#None)


**is_compiling() → [bool](https://docs.python.org/3/library/functions.html#bool)[[source]](../_modules/torch_geometric/_compile.html#is_compiling)**
: Returns [True](https://docs.python.org/3/library/constants.html#True) in case PyTorch is compiling via
`torch.compile()`.


**Return type:**
: [bool](https://docs.python.org/3/library/functions.html#bool)


**is_debug_enabled() → [bool](https://docs.python.org/3/library/functions.html#bool)[[source]](../_modules/torch_geometric/debug.html#is_debug_enabled)**
: Returns [True](https://docs.python.org/3/library/constants.html#True) if the debug mode is enabled.


**Return type:**
: [bool](https://docs.python.org/3/library/functions.html#bool)


***class *debug[[source]](../_modules/torch_geometric/debug.html#debug)**
: Context-manager that enables the debug mode to help track down errors
and separate usage errors from real bugs.


```
with torch_geometric.debug():
    out = model(data.x, data.edge_index)
```


***class *set_debug(*mode: [bool](https://docs.python.org/3/library/functions.html#bool)*)[[source]](../_modules/torch_geometric/debug.html#set_debug)**
: Context-manager that sets the debug mode on or off.


set_debug will enable or disable the debug mode based on its
argument `mode`.
It can be used as a context-manager or as a function.


See debug above for more details.


**is_experimental_mode_enabled(*options: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)]]] = None*) → [bool](https://docs.python.org/3/library/functions.html#bool)[[source]](../_modules/torch_geometric/experimental.html#is_experimental_mode_enabled)**
: Returns [True](https://docs.python.org/3/library/constants.html#True) if the experimental mode is enabled. See
`torch_geometric.experimental_mode` for a list of (optional)
options.


**Return type:**
: [bool](https://docs.python.org/3/library/functions.html#bool)


***class *experimental_mode(*options: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)]]] = None*)[[source]](../_modules/torch_geometric/experimental.html#experimental_mode)**
: Context-manager that enables the experimental mode to test new but
potentially unstable features.


```
with torch_geometric.experimental_mode():
    out = model(data.x, data.edge_index)
```


**Parameters:**
: **options** ([str](https://docs.python.org/3/library/stdtypes.html#str)* or *[list](https://docs.python.org/3/library/stdtypes.html#list)*, **optional*) – Currently there are no experimental
features.


***class *set_experimental_mode(*mode: [bool](https://docs.python.org/3/library/functions.html#bool)*, *options: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[Union](https://docs.python.org/3/library/typing.html#typing.Union)[[str](https://docs.python.org/3/library/stdtypes.html#str), [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)]]] = None*)[[source]](../_modules/torch_geometric/experimental.html#set_experimental_mode)**
: Context-manager that sets the experimental mode on or off.


set_experimental_mode will enable or disable the experimental mode
based on its argument `mode`.
It can be used as a context-manager or as a function.


See experimental_mode above for more details.


**disable_dynamic_shapes(*required_args: [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)]*) → [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[source]](../_modules/torch_geometric/experimental.html#disable_dynamic_shapes)**
: A decorator that disables the usage of dynamic shapes for the given
arguments, i.e., it will raise an error in case `required_args` are
not passed and needs to be automatically inferred.


**Return type:**
: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)


