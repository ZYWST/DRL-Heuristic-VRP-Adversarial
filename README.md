# RERA-VRP-UD

Reactive Routing Agent for Vehicle Routing Problems with Uncertain Demands.

## Overview

This repository implements a Deep Reinforcement Learning (DRL) guided reactive routing framework for solving the Vehicle Routing Problem with uncertain demands (VRP-UD). The approach combines Real-time Adaptive Metaheuristics (RAMA) with DRL-based policy guidance.

## Repository Structure

```
├── data/                   # Problem instances and geographic data
│   ├── problem_instances/  # .dat instance files by scale
│   └── geo_data/           # Geographic network base data
├── src/                    # Core algorithm source code
│   ├── mathematical/       # MILP formulations
│   ├── env/                # DRL environment and configuration
│   ├── algorithms/         # Heuristic solver cores
│   └── utils/              # Utility tools
├── models/                 # Pretrained weights and checkpoints
│   └── pretrained/
├── scripts/                # Experiment execution scripts
│   ├── training/           # Training scripts
│   ├── benchmark/          # Benchmark comparison experiments
│   ├── sensitivity/        # Sensitivity analysis experiments
│   └── ablation/           # Ablation study experiments
└── visualization/          # Plotting and log extraction tools
    ├── evaluation_plots/   # Academic plotting scripts
    ├── log_extractors/     # Log extraction utilities
    └── mapping/            # Map rendering code
```

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Training, benchmarking, sensitivity analysis, and ablation study experiments can be run via scripts in the `scripts/` directory.
