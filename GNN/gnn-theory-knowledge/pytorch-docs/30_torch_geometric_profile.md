| profileit | A decorator to facilitate profiling a function,e.g., obtaining training runtime and memory statistics of a specific model on a specific dataset. |
| --- | --- |
| timeit | A context decorator to facilitate timing a function,e.g., obtaining the runtime of a specific model on a specific dataset. |
| get_stats_summary | Creates a summary of collected runtime and memory statistics. |
| trace_handler |  |
| print_time_total |  |
| rename_profile_file |  |
| torch_profile |  |
| xpu_profile |  |
| count_parameters | Given atorch.nn.Module, count its trainable parameters. |
| get_model_size | Given atorch.nn.Module, get its actual disk size in bytes. |
| get_data_size | Given atorch_geometric.data.Dataobject, get its theoretical memory usage in bytes. |
| get_cpu_memory_from_gc | Returns the used CPU memory in bytes, as reported by thePythongarbage collector. |
| get_gpu_memory_from_gc | Returns the used GPU memory in bytes, as reported by thePythongarbage collector. |
| get_gpu_memory_from_nvidia_smi | Returns the free and used GPU memory in megabytes, as reported bynivdia-smi. |
| get_gpu_memory_from_ipex | Returns the XPU memory statistics. |
| benchmark | Benchmark a list of functionsfuncsthat receive the same set of argumentsargs. |
| nvtxit | Enables NVTX profiling for a function. |


GNN profiling package.


**profileit(*device: [str](https://docs.python.org/3/library/stdtypes.html#str)*)[[source]](../_modules/torch_geometric/profile/profile.html#profileit)**
: A decorator to facilitate profiling a function, *e.g.*, obtaining
training runtime and memory statistics of a specific model on a specific
dataset.
Returns a `GPUStats` if `device` is `xpu` or extended
object `CUDAStats`, if `device` is `cuda`.


**Parameters:**
: **device** ([str](https://docs.python.org/3/library/stdtypes.html#str)) – Target device for profiling. Options are:
`cuda` and obj:xpu.


```
@profileit("cuda")
def train(model, optimizer, x, edge_index, y):
    optimizer.zero_grad()
    out = model(x, edge_index)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return float(loss)

loss, stats = train(model, x, edge_index, y)
```


***class *timeit(*log: [bool](https://docs.python.org/3/library/functions.html#bool) = True*, *avg_time_divisor: [int](https://docs.python.org/3/library/functions.html#int) = 0*)[[source]](../_modules/torch_geometric/profile/profile.html#timeit)**
: A context decorator to facilitate timing a function, *e.g.*, obtaining
the runtime of a specific model on a specific dataset.


```
@torch.no_grad()
def test(model, x, edge_index):
    return model(x, edge_index)

with timeit() as t:
    z = test(model, x, edge_index)
time = t.duration
```


**Parameters:**
: - **log** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [False](https://docs.python.org/3/library/constants.html#False), will not log any runtime
to the console. (default: [True](https://docs.python.org/3/library/constants.html#True))
- **avg_time_divisor** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – If set to a value greater than
`1`, will divide the total time by this value. Useful for
calculating the average of runtimes within a for-loop.
(default: `0`)


**reset()[[source]](../_modules/torch_geometric/profile/profile.html#timeit.reset)**
: Prints the duration and resets current timer.


**get_stats_summary(*stats_list: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[List](https://docs.python.org/3/library/typing.html#typing.List)[GPUStats], [List](https://docs.python.org/3/library/typing.html#typing.List)[CUDAStats]]*) → [Union](https://docs.python.org/3/library/typing.html#typing.Union)[GPUStatsSummary, CUDAStatsSummary][[source]](../_modules/torch_geometric/profile/profile.html#get_stats_summary)**
: Creates a summary of collected runtime and memory statistics.
Returns a `GPUStatsSummary` if list of `GPUStats` was passed,
otherwise (list of `CUDAStats` was passed),
returns a `CUDAStatsSummary`.


**Parameters:**
: **stats_list** (*Union**[**List**[**GPUStats**]**, **List**[**CUDAStats**]**]*) – A list of
`GPUStats` or `CUDAStats` objects, as returned by
profileit().

**Return type:**
: `Union`[`GPUStatsSummary`, `CUDAStatsSummary`]


**trace_handler(*p*)[[source]](../_modules/torch_geometric/profile/profile.html#trace_handler)**
:


**print_time_total(*p*)[[source]](../_modules/torch_geometric/profile/profile.html#print_time_total)**
:


**rename_profile_file(**args*)[[source]](../_modules/torch_geometric/profile/profile.html#rename_profile_file)**
:


**torch_profile(*export_chrome_trace=True*, *csv_data=None*, *write_csv=None*)[[source]](../_modules/torch_geometric/profile/profile.html#torch_profile)**
:


**xpu_profile(*export_chrome_trace=True*)[[source]](../_modules/torch_geometric/profile/profile.html#xpu_profile)**
:


**count_parameters(*model: [Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)*) → [int](https://docs.python.org/3/library/functions.html#int)[[source]](../_modules/torch_geometric/profile/utils.html#count_parameters)**
: Given a [torch.nn.Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module), count its trainable parameters.


**Parameters:**
: **model** (*torch.nn.Model*) – The model.

**Return type:**
: [int](https://docs.python.org/3/library/functions.html#int)


**get_model_size(*model: [Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module)*) → [int](https://docs.python.org/3/library/functions.html#int)[[source]](../_modules/torch_geometric/profile/utils.html#get_model_size)**
: Given a [torch.nn.Module](https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module), get its actual disk size in bytes.


**Parameters:**
: **model** (*torch model*) – The model.

**Return type:**
: [int](https://docs.python.org/3/library/functions.html#int)


**get_data_size(*data: BaseData*) → [int](https://docs.python.org/3/library/functions.html#int)[[source]](../_modules/torch_geometric/profile/utils.html#get_data_size)**
: Given a [torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) object, get its theoretical
memory usage in bytes.


**Parameters:**
: **data** ([torch_geometric.data.Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data)* or *[torch_geometric.data.HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData)) – The [Data](../generated/torch_geometric.data.Data.html#torch_geometric.data.Data) or
[HeteroData](../generated/torch_geometric.data.HeteroData.html#torch_geometric.data.HeteroData) graph object.

**Return type:**
: [int](https://docs.python.org/3/library/functions.html#int)


**get_cpu_memory_from_gc() → [int](https://docs.python.org/3/library/functions.html#int)[[source]](../_modules/torch_geometric/profile/utils.html#get_cpu_memory_from_gc)**
: Returns the used CPU memory in bytes, as reported by the
Python garbage collector.


**Return type:**
: [int](https://docs.python.org/3/library/functions.html#int)


**get_gpu_memory_from_gc(*device: [int](https://docs.python.org/3/library/functions.html#int) = 0*) → [int](https://docs.python.org/3/library/functions.html#int)[[source]](../_modules/torch_geometric/profile/utils.html#get_gpu_memory_from_gc)**
: Returns the used GPU memory in bytes, as reported by the
Python garbage collector.


**Parameters:**
: **device** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The GPU device identifier. (default: `1`)

**Return type:**
: [int](https://docs.python.org/3/library/functions.html#int)


**get_gpu_memory_from_nvidia_smi(*device: [int](https://docs.python.org/3/library/functions.html#int) = 0*, *digits: [int](https://docs.python.org/3/library/functions.html#int) = 2*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)][[source]](../_modules/torch_geometric/profile/utils.html#get_gpu_memory_from_nvidia_smi)**
: Returns the free and used GPU memory in megabytes, as reported by
`nivdia-smi`.


> **Note:** `nvidia-smi` will generally overestimate the amount of memory used
by the actual program, see [here](https://pytorch.org/docs/stable/notes/faq.html#my-gpu-memory-isn-t-freed-properly).


**Parameters:**
: - **device** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The GPU device identifier. (default: `1`)
- **digits** ([int](https://docs.python.org/3/library/functions.html#int)) – The number of decimals to use for megabytes.
(default: `2`)

**Return type:**
: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]


**get_gpu_memory_from_ipex(*device: [int](https://docs.python.org/3/library/functions.html#int) = 0*, *digits=2*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)][[source]](../_modules/torch_geometric/profile/utils.html#get_gpu_memory_from_ipex)**
: Returns the XPU memory statistics.


**Parameters:**
: - **device** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The GPU device identifier. (default: `0`)
- **digits** ([int](https://docs.python.org/3/library/functions.html#int)) – The number of decimals to use for megabytes.
(default: `2`)

**Return type:**
: [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float), [float](https://docs.python.org/3/library/functions.html#float)]


**benchmark(*funcs: [List](https://docs.python.org/3/library/typing.html#typing.List)[[Callable](https://docs.python.org/3/library/typing.html#typing.Callable)]*, *args: [Union](https://docs.python.org/3/library/typing.html#typing.Union)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)], [List](https://docs.python.org/3/library/typing.html#typing.List)[[Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)]]]*, *num_steps: [int](https://docs.python.org/3/library/functions.html#int)*, *func_names: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/library/stdtypes.html#str)]] = None*, *num_warmups: [int](https://docs.python.org/3/library/functions.html#int) = 10*, *backward: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *per_step: [bool](https://docs.python.org/3/library/functions.html#bool) = False*, *progress_bar: [bool](https://docs.python.org/3/library/functions.html#bool) = False*)[[source]](../_modules/torch_geometric/profile/benchmark.html#benchmark)**
: Benchmark a list of functions `funcs` that receive the same set
of arguments `args`.


**Parameters:**
: - **funcs** (*[**Callable**]*) – The list of functions to benchmark.
- **args** (*(**Any**, **) or **[**(**Any**, **)**]*) – The arguments to pass to the functions.
Can be a list of arguments for each function in `funcs` in
case their headers differ.
Alternatively, you can pass in functions that generate arguments
on-the-fly (e.g., useful for benchmarking models on various sizes).
- **num_steps** ([int](https://docs.python.org/3/library/functions.html#int)) – The number of steps to run the benchmark.
- **func_names** (*[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]**, **optional*) – The names of the functions. If not given,
will try to infer the name from the function itself.
(default: [None](https://docs.python.org/3/library/constants.html#None))
- **num_warmups** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – The number of warmup steps.
(default: `10`)
- **backward** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will benchmark both
forward and backward passes. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **per_step** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will report runtimes
per step. (default: [False](https://docs.python.org/3/library/constants.html#False))
- **progress_bar** ([bool](https://docs.python.org/3/library/functions.html#bool)*, **optional*) – If set to [True](https://docs.python.org/3/library/constants.html#True), will print a
progress bar during benchmarking. (default: [False](https://docs.python.org/3/library/constants.html#False))


**nvtxit(*name: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[str](https://docs.python.org/3/library/stdtypes.html#str)] = None*, *n_warmups: [int](https://docs.python.org/3/library/functions.html#int) = 0*, *n_iters: [Optional](https://docs.python.org/3/library/typing.html#typing.Optional)[[int](https://docs.python.org/3/library/functions.html#int)] = None*)[[source]](../_modules/torch_geometric/profile/nvtx.html#nvtxit)**
: Enables NVTX profiling for a function.


**Parameters:**
: - **name** (*Optional**[*[str](https://docs.python.org/3/library/stdtypes.html#str)*]**, **optional*) – Name to give the reference frame for
the function being wrapped. Defaults to the name of the
function in code.
- **n_warmups** ([int](https://docs.python.org/3/library/functions.html#int)*, **optional*) – Number of iters to call that function
before starting. Defaults to 0.
- **n_iters** (*Optional**[*[int](https://docs.python.org/3/library/functions.html#int)*]**, **optional*) – Number of iters of that function to
record. Defaults to all of them.


