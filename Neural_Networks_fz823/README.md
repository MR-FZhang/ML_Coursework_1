# Machine Learning Neural Network Coursework

This repository contains the starter code and datasets for the Neural Networks coursework.

## Structure

* `part1_nn_lib.py`: Implementation of a modular neural network library from scratch using NumPy.
  - Implement the layer operations (`forward`, `backward`, `update_params`) for `SigmoidLayer`, `ReluLayer`, `LinearLayer`, and `MultiLayerNetwork`.
  - Implement data normalization and reversal in `Preprocessor`.
  - Supported activations in `MultiLayerNetwork` include `"sigmoid"`, `"relu"`, and `"identity"` (linear passthrough without activation).
* `part2_house_value_regression.py`: Implementation of a neural network regressor for the California Housing dataset using PyTorch.
  - Implement `fit`, `predict`, and `evaluate` methods in `Regressor`.
  - Train and save your best model to `part2_model.pickle` using `save_regressor()`.
* `housing.csv`: California Housing dataset used for Part 2.
* `iris.dat`: Iris dataset used for the example workflow in Part 1.
* `requirements.txt`: Python package dependencies for the coursework.

## Setup

Ensure your Python environment is set up with the required dependencies:
```bash
pip install -r requirements.txt
```
If working on the lab machines, you can use the pre-configured virtual environment:
```bash
source /vol/lab/ml/mlvenv/bin/activate
```

## Submission & Testing

1. Test your implementation using LabTS.
2. For Part 2, ensure your final trained model is saved as `part2_model.pickle` and committed to your repository.
3. Submit your final commit via the LabTS interface before the deadline.
