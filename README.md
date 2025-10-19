# quant-feature-toolkit  
A Python research toolkit for building and testing scale-invariant trading indicators.

## Overview  
This repository contains a collection of feature-engineering modules designed for log-price-based time series, allowing cross-asset comparability and regime adaptability.  
It includes:  
- A library of indicators (e.g., **Entropy**, **CMMA**, **FTI**, etc.) implemented in `indicators/`.  
- A lightweight testing and research environment in `research/`, used to verify that indicators produce correct and stable values on synthetic datasets.

---

## Folder Structure
```quant-feature-toolkit/
├── indicators/ # Core indicator implementations
│ ├── cmma.py
│ ├── entropy.py
│ ├── fti.py
│ ├── ...
│
├── research/ # Testing environment using synthetic data
│ ├── test_indicators.py # Verifies indicator output correctness and stability
│ ├── utils.py # Helper functions for generating test signals
│ └── ...
│
├── README.md
```
About the research/ Folder

The research/ directory is not used for production trading code.
Instead, it provides:

Synthetic test data (random-walk style price series) to validate each indicator’s logic.

Smoke tests and sanity checks that confirm outputs have the expected shape, scale, and numerical stability.

Exploratory scripts for debugging or plotting indicator behavior before integration.

This ensures that all indicators in the indicators/ library generate consistent, reproducible results before being used in larger backtesting or modeling pipelines.
