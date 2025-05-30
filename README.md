# Symmetry-Aware GFlowNets
---
Official implementation of ICML 2025 paper "Symmetry-Aware GFlowNets".


## Environment
- networkx
- igraph
- matplotlib
- rdkit
- torch
- torch_scatter
- torch-geometric
- numpy
- scipy
- scikit-learn

## Code References
Much of our implementation, as well as the molecule generation experiments in our paper, are based on https://github.com/recursionpharma/gflownet.


## Experiments

Experiments can be run using the following command:

```python main.py [config_file.yaml] [logging_path]```

We included two sample configuration files in `config` directory, which were used for the experiments in the paper. To reproduce the results, set the `correction_method` option to one of the following: "vanilla", "transition", "pe", "flowscaling" or "rewardscaling".
