PyG is available for Python 3.10 to Python 3.13.


> **Note:** We do not recommend installation as a root user on your system Python.
Please setup a virtual environment, *e.g.*, via [venv](https://virtualenv.pypa.io/en/latest),  [Anaconda/Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install), or create a [Docker image](https://www.docker.com/).


## Quick Start


PyTorch
Your OS
Package
CUDA
Run:


```

```


## Installation via PyPI


From  **PyG 2.3** onwards, you can install and use PyG **without any external library** required except for PyTorch.
For this, simply run:


```
pip install torch_geometric
```


### Additional Libraries


If you want to utilize the full set of features from PyG, there exists several additional libraries you may want to install:


- [pyg-lib](https://github.com/pyg-team/pyg-lib): Heterogeneous GNN operators and graph sampling routines
- [torch-scatter](https://github.com/rusty1s/pytorch_scatter): Accelerated and efficient sparse reductions
- [torch-sparse](https://github.com/rusty1s/pytorch_sparse): `SparseTensor` support, see [here](https://pytorch-geometric.readthedocs.io/en/latest/advanced/sparse_tensor.html)
- [torch-cluster](https://github.com/rusty1s/pytorch_cluster): Graph clustering routines
- [torch-spline-conv](https://github.com/rusty1s/pytorch_spline_conv): [SplineConv](../generated/torch_geometric.nn.conv.SplineConv.html#torch_geometric.nn.conv.SplineConv) support


These packages come with their own CPU and GPU kernel implementations based on the  [PyTorch C++/CUDA/hip(ROCm) extension interface](https://github.com/pytorch/extension-cpp/).
For a basic usage of PyG, these dependencies are **fully optional**.
We recommend to start with a minimal installation, and install additional dependencies once you start to actually need them.


### Installation from Wheels


For ease of installation of these extensions, we provide `pip` wheels for these packages for all major OS, PyTorch and CUDA combinations, see [here](https://data.pyg.org/whl):


1. Ensure that at least PyTorch 1.13.0 is installed:


```
python -c "import torch; print(torch.__version__)"
>>> 2.6.0
```
2. Find the CUDA version PyTorch was installed with:


```
python -c "import torch; print(torch.version.cuda)"
>>> 12.6
```
3. Install the relevant packages:


```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```


where `${TORCH}` and `${CUDA}` should be replaced by the specific PyTorch and CUDA versions, respectively:


- PyTorch 2.8.*: `${TORCH}=2.8.0` and `${CUDA}=cpu|cu126|cu128|cu129`
- PyTorch 2.7.*: `${TORCH}=2.7.0` and `${CUDA}=cpu|cu118|cu126|cu128`
- PyTorch 2.6.*: `${TORCH}=2.6.0` and `${CUDA}=cpu|cu118|cu124|cu126`
- PyTorch 2.5.*: `${TORCH}=2.5.0` and `${CUDA}=cpu|cu118|cu121|cu124`
- PyTorch 2.4.*: `${TORCH}=2.4.0` and `${CUDA}=cpu|cu118|cu121|cu124`
- PyTorch 2.3.*: `${TORCH}=2.3.0` and `${CUDA}=cpu|cu118|cu121`
- PyTorch 2.2.*: `${TORCH}=2.2.0` and `${CUDA}=cpu|cu118|cu121`
- PyTorch 2.1.*: `${TORCH}=2.1.0` and `${CUDA}=cpu|cu118|cu121`
- PyTorch 2.0.*: `${TORCH}=2.0.0` and `${CUDA}=cpu|cu117|cu118`
- PyTorch 1.13.*: `${TORCH}=1.13.0` and `${CUDA}=cpu|cu116|cu117`


For example, for PyTorch 2.8.* and CUDA 12.9, type:


```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu129.html
```


For example, for PyTorch 2.6.* and CUDA 12.6, type:


```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
```


**Note:** Binaries of older versions are also provided for PyTorch 1.4.0, 1.5.0, 1.6.0, 1.7.0/1.7.1, 1.8.0/1.8.1, 1.9.0, 1.10.0/1.10.1/1.10.2, 1.11.0, 1.12.0/1.12.1, 1.13.0/1.13.1, 2.0.0/2.0.1, 2.1.0/2.1.1/2.1.2, 2.2.0/2.2.1/2.2.2, 2.3.0/2.3.1, 2.4.0/2.4.1, and 2.5.0/2.5.1 (following the same procedure).
**For older versions, you need to explicitly specify the latest supported version number** or install via `pip install --no-index` in order to prevent a manual installation from source.
You can look up the latest supported version number [here](https://data.pyg.org/whl).


**ROCm:** The external [pyg-rocm-build repository](https://github.com/Looong01/pyg-rocm-build) provides wheels and detailed instructions on how to install PyG for ROCm.
If you have any questions about it, please open an issue [here](https://github.com/Looong01/pyg-rocm-build/issues).


### Installation from Source


In case a specific version is not supported by [our wheels](https://data.pyg.org/whl), you can alternatively install them from source:


1. Ensure that your CUDA is setup correctly (optional):


1. Check if PyTorch is installed with CUDA support:


```
python -c "import torch; print(torch.cuda.is_available())"
>>> True
```
2. Add CUDA to `$PATH` and `$CPATH` (note that your actual CUDA path may vary from `/usr/local/cuda`):


```
export PATH=/usr/local/cuda/bin:$PATH
echo $PATH
>>> /usr/local/cuda/bin:...

export CPATH=/usr/local/cuda/include:$CPATH
echo $CPATH
>>> /usr/local/cuda/include:...
```
3. Add CUDA to `$LD_LIBRARY_PATH` on Linux and to `$DYLD_LIBRARY_PATH` on macOS (note that your actual CUDA path may vary from `/usr/local/cuda`):


```
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
>>> /usr/local/cuda/lib64:...

export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH
echo $DYLD_LIBRARY_PATH
>>> /usr/local/cuda/lib:...
```
4. Verify that `nvcc` is accessible from terminal:


```
nvcc --version
>>> 11.8
```
5. Ensure that PyTorch and system CUDA versions match:


```
python -c "import torch; print(torch.version.cuda)"
>>> 11.8

nvcc --version
>>> 11.8
```
2. Install the relevant packages:


```
pip install --verbose git+https://github.com/pyg-team/pyg-lib.git
pip install --verbose torch_scatter
pip install --verbose torch_sparse
pip install --verbose torch_cluster
pip install --verbose torch_spline_conv
```


In rare cases, CUDA or Python path problems can prevent a successful installation.
`pip` may even signal a successful installation, but execution simply crashes with `Segmentation fault (core dumped)`.
We collected common installation errors in the [Frequently Asked Questions](installation.html#frequently-asked-questions) subsection.
In case the FAQ does not help you in solving your problem, please create an [issue](https://github.com/pyg-team/pytorch_geometric/issues).
Before, please verify that your CUDA is set up correctly by following the official [installation guide](https://docs.nvidia.com/cuda).


## Installation via Anaconda


> **Warning:** Conda packages are no longer available since PyTorch `>2.5.0`.
Please use `pip` instead.


For earlier PyTorch versions (`torch<=2.5.0`), you can install PyG via  [Anaconda](https://anaconda.org/pyg/pyg) for all major OS, and CUDA combinations.
If you have not yet installed PyTorch, install it via  `conda install` as described in its [official documentation](https://pytorch.org/get-started/locally/).
Given that you have PyTorch installed, run


```
conda install pyg -c pyg
```


If  `conda` does not pick up the correct CUDA version of PyG, you can enforce it as follows:


```
conda install pyg=*=*cu* -c pyg
```


## Enabling Accelerated cuGraph GNNs


Currently, NVIDIA recommends NVIDIA PyG Container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pyg/tags>_ to use cuGraph integration in PyG.
This functionality is planned to be enabled through cuDNN which is part of PyTorch builds. We still recommend using the NVIDIA PyG Container regardless to have the fastest and most stable build of the NVIDIA CUDA stack combined with PyTorch and PyG.


## Frequently Asked Questions


1. `undefined symbol: **make_function_schema**`: This issue signals (1) a **version conflict** between your installed PyTorch version and the `${TORCH}` version specified to install the extension packages, or (2) a version conflict between the installed CUDA version of PyTorch and the `${CUDA}` version specified to install the extension packages.
Please verify that your PyTorch version and its CUDA version **match** with your installation command:


```
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```


For re-installation, ensure that you do not run into any caching issues by using the `pip --force-reinstall --no-cache-dir` flags.
In addition, the `pip --verbose` option may help to track down any issues during installation.
If you still do not find any success in installation, please try to install the extension packages [from source](installation.html#installation-from-source).


