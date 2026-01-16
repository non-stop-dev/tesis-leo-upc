import os
import shutil

# Target directory
base_dir = "/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/GNN/gnn-theory-knowledge/pytorch-docs"

# Define the ordered list and their mapping to current filenames
# Mapping is based on the previously generated filenames
ordered_mapping = [
    ("install_installation.md", "01_installation.md"),
    ("get_started_introduction.md", "02_introduction_by_example.md"),
    ("get_started_colabs.md", "03_colab_notebooks_and_video_tutorials.md"),
    ("tutorial_gnn_design.md", "04_design_of_graph_neural_networks.md"),
    ("tutorial_datase.md", "05_working_with_graph_datasets.md"),
    ("tutorial_application.md", "06_use_cases_and_applications.md"),
    ("tutorial_distributed.md", "07_distributed_training.md"),
    ("advanced_batching.md", "08_advanced_mini_batching.md"),
    ("advanced_sparse_tensor.md", "09_memory_efficient_aggregations.md"),
    ("advanced_hga.md", "10_hierarchical_neighborhood_sampling.md"),
    ("advanced_compile.md", "11_compiled_graph_neural_networks.md"),
    ("advanced_ji.md", "12_torchscript_support.md"),
    ("advanced_remote.md", "13_scaling_up_gnns_via_remote_backends.md"),
    ("advanced_graphgy.md", "14_managing_experiments_with_graphgym.md"),
    ("advanced_cpu_affinity.md", "15_cpu_affinity_for_pyg_workloads.md"),
    ("modules_roo.md", "16_torch_geometric.md"),
    ("modules_nn.md", "17_torch_geometric_nn.md"),
    ("modules_data.md", "18_torch_geometric_data.md"),
    ("modules_loader.md", "19_torch_geometric_loader.md"),
    ("modules_sampler.md", "20_torch_geometric_sampler.md"),
    ("modules_datasets.md", "21_torch_geometric_datasets.md"),
    ("modules.md", "22_torch_geometric_llm.md"),
    ("modules_transforms.md", "23_torch_geometric_transforms.md"),
    ("modules_utils.md", "24_torch_geometric_utils.md"),
    ("modules_explain.md", "25_torch_geometric_explain.md"),
    ("modules_metrics.md", "26_torch_geometric_metrics.md"),
    ("modules_distributed.md", "27_torch_geometric_distributed.md"),
    ("modules_contrib.md", "28_torch_geometric_contrib.md"),
    ("modules_graphgy.md", "29_torch_geometric_graphgym.md"),
    ("modules_profile.md", "30_torch_geometric_profile.md"),
    ("cheatsheet_gnn_cheatshee.md", "31_gnn_cheatsheet.md"),
    ("cheatsheet_data_cheatshee.md", "32_dataset_cheatsheet.md"),
]

# Files I added that were not in the user's explicit list but provide the CONTENT for tutorials
extra_mapping = [
    ("tutorial_create_gnn.md", "04a_creating_message_passing_networks.md"),
    ("tutorial_heterogeneous.md", "04b_heterogeneous_graph_learning.md"),
    ("tutorial_load_csv.md", "05a_loading_graphs_from_csv.md"),
    ("tutorial_neighbor_loader.md", "05b_neighbor_sampling.md"),
    ("tutorial_explain.md", "06a_gnn_explainability.md"),
    ("tutorial_point_cloud.md", "06b_point_cloud_processing.md"),
]

# Execute renaming
print("Starting file renaming...")

# Rename the core list
for current, new_name in ordered_mapping:
    old_path = os.path.join(base_dir, current)
    new_path = os.path.join(base_dir, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"Renamed: {current} -> {new_name}")
    else:
        # Check if it was already renamed (avoid error on rerun)
        if os.path.exists(new_path):
             print(f"Already exists: {new_name}")
        else:
             print(f"File not found: {current}")

# Rename the extras
for current, new_name in extra_mapping:
    old_path = os.path.join(base_dir, current)
    new_path = os.path.join(base_dir, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"Renamed Extra: {current} -> {new_name}")

# Move index to 00
if os.path.exists(os.path.join(base_dir, "00_index.md")):
    print("00_index.md already exists.")
elif os.path.exists(os.path.join(base_dir, "index.md")):
    os.rename(os.path.join(base_dir, "index.md"), os.path.join(base_dir, "00_index.md"))
    print("Renamed index.md -> 00_index.md")

print("Renaming completed.")
