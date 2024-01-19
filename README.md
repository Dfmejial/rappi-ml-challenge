
# Rappi Challenge

The **app-rappi-dfmejial** is a Python package that provides a workflow for handling the Rappi Challenge on the Titanic dataset. The package includes functionality for reading raw data, preprocessing it, training a machine learning model (Random Forest Classifier), and evaluating the model's performance.

## Installation

You can install the **app-rappi-dfmejial** package using `pip`. Open a terminal and run the following command:

```bash
pip install app-rappi-dfmejial
```

# Usage
Command Line Interface (CLI)
The package includes a command-line interface for running the Rappi Challenge workflow. You can execute it with the following command:

```bash
rappi-dfmejial --estimators <number_of_estimators> --depth <max_depth> [--save-model]
```
```bash
--estimators: Number of trees to use in the random forest model (default is 500).
--depth: Maximum depth of each tree in the random forest model (default is 20).
--save-model: Optional flag indicating whether to save the trained model.
```

# Python API
The Rappi Challenge can also be used as a Python module. Here's an example of how to use it:

```python
from app_rappi_dfmejial import RappiChallenge

# Create a RappiChallenge instance with custom parameters
rappi_challenge = RappiChallenge(n_estimators=500, max_depth=20, save_model=True)

# Run the challenge workflow
model_path = rappi_challenge.run_challenge()

# The trained model is saved at the specified path
print(f"Trained model saved at: {model_path}")
```

# Dependencies

Python 3.9 or higher

Other dependencies are automatically installed during package installation.

