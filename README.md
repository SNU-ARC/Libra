# Libra: Effective yet Efficient Load Balancing for Large-scale MoE Inference

Libra is a system for MoE inference that achieves near-optimal expert load balancing with minimal overhead by predicting future expert activations and restructuring execution to overlap load-balancing costs with MoE computation. This repository provides an implementation of Libra atop SGLang.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)

## Overview
Libra is a dynamic load balancing system for distributed MoE inference under expert parallelism, designed to mitigate stragglers caused by expert load imbalance.
Libra uses a lookahead predictor to anticipate next-layer expert activations and plan locality-aware expert replication.
By restructuring execution (Two-Stage Locality-Aware Execution), Libra overlaps sophisticated load-balancing—such as adaptive token sharding applied only to remote tokens and offloaded to the CPU—with MoE computation to minimize overhead.

## Features
- **Two-Stage Locality-Aware Execution**: Splits MoE computation into two phases based on token locality.
  - MoE<sub>local</sub>: Processes token routed to experts residing on the same GPU as the tokens themselves.
  - MoE<sub>remote</sub>: Handles tokens that must be dispatched to other GPUs.
- **Hot Expert Replication**: Introduces an additional refinement to hot expert replication.
  - Lookahead Predictor: Employs a lookahead predictor, leveraging a well-established property that hidden states in Transformer-based LLMs evolve slowly across layers.
  - Locality-Aware Expert Replication Planning: Performs expert replication planning in two phases. In the first phase, each GPU imports the top N × α most frequently activated non-resident experts for its local tokens, extending the MoE<sub>local</sub> computation window. In the second phase, the hottest expert from the most heavily loaded GPU is iteratively replicated to the least-loaded eligible GPU until each GPU hosts up to N additional experts, ensuring load balancing.
- **Adaptive Token Sharding**: Dynamically balances workload by iteratively redistributing remote tokens from overloaded GPUs to underutilized GPUs that host the corresponding expert replicas.

## Prerequisites
- Python 3.10
- CUDA 12.6+
- GPUs with Hopper architecture or newer 
- NVLink or NVSwitch interconnect

## Installation

### 1. Clone Libra Repository
Clone the main repository along with its submodules (`Libra-Core`, `Libra-Internal`) in one go:

```bash
git clone --recursive [https://github.com/SNU-ARC/Libra.git](https://github.com/SNU-ARC/Libra.git)
cd Libra
```

### 2. Create Environments and Install Dependencies

You need to set up three separate environments for Libra, Lina, and Libra Internal.

```bash
# Environment setup - Libra
conda create -n libra python=3.10 -y
conda activate libra
cd Libra-Core
git checkout iclr2026_libra
pip install -e "python[all]"
cd ..

# Environment setup - Lina
conda create -n lina python=3.10 -y
conda activate lina
cd Libra-Core
git checkout iclr2026_lina
pip install -e "python[all]"
cd ..

# Environment setup - Libra Internal (required for imbalance ratio measurement)
conda create -n libra_internal python=3.10 -y
conda activate libra_internal
cd Libra-Internal
git checkout iclr2026_libra
pip install -e "python[all]"
cd ..
```

## Usage

### Libra
```bash
# Ensure you are on the libra branch
cd Libra-Core
git checkout iclr2026_libra

bash single_node_scripts/ep_test.sh ${MODEL} ${SEQ_LEN} ${MINI_BATCH_SIZE} EP ${DATASET} train ${START_PORTION} ${END_PORTION} ${START_IDX} ${END_IDX} ${N} ${L} ${SEQ_LENS_SUM} ${SCHEME}

# Usage Example
# bash ./single_node_scripts/ep_test.sh Qwen3-235B-A22B 1024 2 EP bookcorpus train 0 1 800 1000 6 4 2048 libra
```
- MODEL: Model to load and run
- SEQ_LEN: Sequence length
- MINI_BATCH_SIZE: Mini batch size
- DATASET: Dataset to use. Downloading from huggingface may be required.
- START_PORTION: Start portion of the dataset
- END_PORTION: End portion of the dataset
- START_IDX: Start index of the dataset used for evaluation
- END_IDX: End idx of the dataset used for evaluation
- N: The number of prefetched experts per GPU
- L: The number of local hot experts per GPU
- SEQ_LENS_SUM: SEQ_LEN * MINI_BATCH_SIZE
- SCHEME: Scheme to apply
- **CAUTION**: Check variable `is_ori` before running the script.
  - For SGLang and EPLB scheme, `is_ori` should be `True`
  - For Libra, `is_ori` should be `False`

### Configuration - Lina
```bash
# Ensure you are on the lina branch
cd Libra-Core
git checkout iclr2026_lina

# Run Qwen3MoE
bash single_node_scripts/ep_test_qwen3.sh ${MODEL} ${SEQ_LEN} ${MINI_BATCH_SIZE} EP ${DATASET} train ${START_PORTION} ${END_PORTION} ${START_IDX} ${END_IDX} ${N} ${SEQ_LENS_SUM}

# Usage Example
# bash ./single_node_scripts/ep_test_qwen3.sh Qwen3-235B-A3B 1024 2 EP bookcorpus train 0 1 800 1000 6 2048

# Run GLM-4.5
bash single_node_scripts/ep_test_glm45.sh ${MODEL} ${SEQ_LEN} ${MINI_BATCH_SIZE} EP ${DATASET} train ${START_PORTION} ${END_PORTION} ${START_IDX} ${END_IDX} ${N} ${SEQ_LENS_SUM}

# Usage Example
# bash ./single_node_scripts/ep_test_glm45.sh GLM-4.5 1024 2 EP bookcorpus train 0 1 800 1000 6 2048
```
- **CAUTION**: Check variable `is_ori` before running the script.
  - For Lina, `is_ori` should be `False`
