# Stochastic Impute: Synthetic Data Generation Engine

A Python utility designed to generate **synthetic population data** and perform **probabilistic imputation** using aggregated public data sources (e.g., from [Stats NZ Data Explorer](https://explore.data.stats.govt.nz)).

This tool is optimized for large-scale microdata generation where individual attributes are assigned based on conditional probability distributions.

---

## 🚀 Key Features

* **Dynamic Column Matching**: Automatically identifies shared features between the base population and reference data.
* **Missingness-Aware Logic**: Handles rows with `NaN` values by dynamically re-calculating probabilities based only on the available non-null features.
* **Stochastic Selection**: Uses weighted random sampling to preserve the natural variance and distribution of the source data.
* **Dual-Source Integration**: If a target attribute already exists, the engine can blend the existing data with the synthetic prediction using row-wise averaging.
* **Optimized Performance**: Processes millions rows in seconds by grouping identical "missingness patterns" rather than iterating row-by-row.

---

## 📋 Data Requirements

### 1. Population Seed (`pop_data`)
The starting point for your synthetic data. 
* Can contain existing columns you wish to "refine."
* Can contain `NaN` values; the engine will ignore these specific columns during matching for those specific rows.

### 2. Reference Distributions (`data_dict`)
Each entry in the dictionary must be a DataFrame containing:
* **Shared Features**: (e.g., `age`, `location`) to match against the seed.
* **Target Column**: The attribute being generated.
* **Probability Column**: A numeric column named **`probability`** representing the weight of that occurrence.

---

## 💻 Example Usage

```python
import pandas as pd
from process.model import stochastic_impute

# 1. Prepare base population
pop = pd.DataFrame({
    'gender': [1, 2, 1],
    'age': [25, 30, np.nan] # Age is missing for the last person
})

# 2. Define reference data (e.g., Work Status distribution)
work_ref = pd.DataFrame({
    'gender': [1, 1, 2, 2],
    'work_status': [1, 2, 1, 2],
    'probability': [0.8, 0.2, 0.6, 0.4]
})

data_sources = {"work_task": work_ref}

# 3. Define the configuration
tasks = {
    "work_task": {
        "targets": ["work_status"]
    }
}

# 4. Execute
synthetic_data = stochastic_impute(
    pop_data=pop,
    data_dict=data_sources,
    task_list=tasks,
    output_dir="./synthetic_output"
)
```