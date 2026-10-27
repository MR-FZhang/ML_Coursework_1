# Changes from the original template

Every change made to the provided skeleton, section by section. For the *why* behind each piece of code, see `IMPLEMENTATION_NOTES.md`.

**See the exact diff yourself.** The repo's only commit, `a5db283 provided code`, is the untouched template, so git shows precisely what changed:

```bash
cd Coursework_1/Neural_Networks_fz823
git diff                       # everything
git diff -- part1_nn_lib.py    # one file
```

**Legend**
- ✅ **Inside a `START/END OF YOUR CODE` block**: the places the template asks you to fill in.
- ⚠️ **Outside a code block**: template structure was touched. These are the ones to double-check against the spec. All of them are allowed: nothing is renamed, and every new argument has a default.

In the "after" snippets below, explanatory comments are left out to keep them short. The real files have them.

---

## Summary

| # | File | Section | Where | Change |
|---|---|---|---|---|
| 1 | part1 | `SigmoidLayer.forward` | ✅ | `pass` → sigmoid + cache output |
| 2 | part1 | `SigmoidLayer.backward` | ✅ | `pass` → `grad_z * s * (1 - s)` |
| 3 | part1 | `ReluLayer.forward` | ✅ | `pass` → `max(0, x)` + cache input |
| 4 | part1 | `ReluLayer.backward` | ✅ | `pass` → `grad_z * (x > 0)` |
| 5 | part1 | `LinearLayer.__init__` | ✅ | `None` → Xavier `W`, zero `b` |
| 6 | part1 | `LinearLayer.forward` | ✅ | `pass` → `x @ W + b` + cache input |
| 7 | part1 | `LinearLayer.backward` | ✅ | `pass` → the 3 gradients |
| 8 | part1 | `LinearLayer.update_params` | ✅ | `pass` → gradient-descent step |
| 9 | part1 | `MultiLayerNetwork.__init__` | ✅ | `None` → build the layer list |
| 10 | part1 | `MultiLayerNetwork.forward` | ✅ | dummy zeros → loop forward |
| 11 | part1 | `MultiLayerNetwork.backward` | ✅ | `pass` → loop in reverse |
| 12 | part1 | `MultiLayerNetwork.update_params` | ✅ | `pass` → update every layer |
| 13 | part1 | `Preprocessor.__init__` | ✅ | `pass` → store column min / range |
| 14 | part1 | `Preprocessor.apply` | ✅ | `pass` → min-max scale |
| 15 | part1 | `Preprocessor.revert` | ✅ | `pass` → inverse scale |
| 16 | part2 | imports | ⚠️ | 3 imports added |
| 17 | part2 | `Regressor.__init__` signature | ⚠️ | `nb_epoch` default 1000 → 150; 3 optional args added |
| 18 | part2 | `Regressor.__init__` body | ✅ | build the PyTorch model |
| 19 | part2 | `Regressor.fit` | ✅ | training loop added |
| 20 | part2 | `Regressor.predict` | ✅ | `pass` → predict + un-scale |
| 21 | part2 | `Regressor.evaluate` | ✅ | `return 0` → RMSE |
| 22 | part2 | `main()` | ⚠️ | train/validation split, report both RMSEs |
| 23 | part2 | `if __name__ == "__main__":` | ⚠️ | call `main()` through an import |
| 24 | repo | `part2_model.pickle` | new file | the trained model (a required deliverable) |

**Not changed at all.**
In `part1_nn_lib.py`: `xavier_init`, `Layer`, `MSELossLayer`, `CrossEntropyLossLayer`, `save_network`, `load_network`, `Trainer`, `example_main`.
In `part2_house_value_regression.py`: `_preprocessor`, `_revert_preprocessor`, `save_regressor`, `load_regressor`.
No docstrings were edited, except that new argument lines were added to `Regressor.__init__`'s docstring (#17).

---

## Part 1: `part1_nn_lib.py`

### 1. `SigmoidLayer.forward` ✅ ([part1_nn_lib.py:122](Neural_Networks_fz823/part1_nn_lib.py:122))
```python
# before
pass
# after
self._cache_current = 1 / (1 + np.exp(-x))
return self._cache_current
```

### 2. `SigmoidLayer.backward` ✅ ([part1_nn_lib.py:148](Neural_Networks_fz823/part1_nn_lib.py:148))
```python
# before
pass
# after
s = self._cache_current
return grad_z * s * (1 - s)
```

### 3. `ReluLayer.forward` ✅ ([part1_nn_lib.py:185](Neural_Networks_fz823/part1_nn_lib.py:185))
```python
# before
pass
# after
self._cache_current = x
return np.maximum(0, x)
```

### 4. `ReluLayer.backward` ✅ ([part1_nn_lib.py:210](Neural_Networks_fz823/part1_nn_lib.py:210))
```python
# before
pass
# after
return grad_z * (self._cache_current > 0)
```

### 5. `LinearLayer.__init__` ✅ ([part1_nn_lib.py:243](Neural_Networks_fz823/part1_nn_lib.py:243))
```python
# before
self._W = None # Need to initialise this
self._b = None # Need to initialise this
# after
self._W = xavier_init((n_in, n_out))
self._b = np.zeros(n_out)
```
The three `_cache_current` / `_grad_*_current = None` lines below were already in the template and are unchanged.
⚠️ The bias shape `(n_out,)` is a judgment call. See section 3 of the notes if a LabTS test complains.

### 6. `LinearLayer.forward` ✅ ([part1_nn_lib.py:272](Neural_Networks_fz823/part1_nn_lib.py:272))
```python
# before
pass
# after
self._cache_current = x
return x @ self._W + self._b
```

### 7. `LinearLayer.backward` ✅ ([part1_nn_lib.py:301](Neural_Networks_fz823/part1_nn_lib.py:301))
```python
# before
pass
# after
self._grad_W_current = self._cache_current.T @ grad_z
self._grad_b_current = grad_z.sum(axis=0)
return grad_z @ self._W.T
```

### 8. `LinearLayer.update_params` ✅ ([part1_nn_lib.py:324](Neural_Networks_fz823/part1_nn_lib.py:324))
```python
# before
pass
# after
self._W = self._W - learning_rate * self._grad_W_current
self._b = self._b - learning_rate * self._grad_b_current
```

### 9. `MultiLayerNetwork.__init__` ✅ ([part1_nn_lib.py:366](Neural_Networks_fz823/part1_nn_lib.py:366))
```python
# before
self._layers = None # Need to initialise this
# after  (self._layers = [] above the block is the template's own line)
activation_classes = {"relu": ReluLayer, "sigmoid": SigmoidLayer}
n_in = input_dim
for n_out, activation in zip(neurons, activations):
    self._layers.append(LinearLayer(n_in, n_out))
    if activation in activation_classes:
        self._layers.append(activation_classes[activation]())
    elif activation != "identity":
        raise ValueError(f"Unknown activation: {activation}")
    n_in = n_out
```

### 10. `MultiLayerNetwork.forward` ✅ ([part1_nn_lib.py:399](Neural_Networks_fz823/part1_nn_lib.py:399))
```python
# before
return np.zeros((1, self.neurons[-1])) # Dummy placeholder. Replace with your own code
# after
for layer in self._layers:
    x = layer(x)
return x
```

### 11. `MultiLayerNetwork.backward` ✅ ([part1_nn_lib.py:429](Neural_Networks_fz823/part1_nn_lib.py:429))
```python
# before
pass
# after
for layer in reversed(self._layers):
    grad_z = layer.backward(grad_z)
return grad_z
```

### 12. `MultiLayerNetwork.update_params` ✅ ([part1_nn_lib.py:451](Neural_Networks_fz823/part1_nn_lib.py:451))
```python
# before
pass
# after
for layer in self._layers:
    layer.update_params(learning_rate)
```

### 13. `Preprocessor.__init__` ✅ ([part1_nn_lib.py:633](Neural_Networks_fz823/part1_nn_lib.py:633))
```python
# before
pass
# after
self._min = data.min(axis=0)
data_range = data.max(axis=0) - self._min
self._range = np.where(data_range == 0, 1, data_range)
```

### 14. `Preprocessor.apply` ✅ ([part1_nn_lib.py:658](Neural_Networks_fz823/part1_nn_lib.py:658))
```python
# before
pass
# after
return (data - self._min) / self._range
```

### 15. `Preprocessor.revert` ✅ ([part1_nn_lib.py:677](Neural_Networks_fz823/part1_nn_lib.py:677))
```python
# before
pass
# after
return data * self._range + self._min
```

---

## Part 2: `part2_house_value_regression.py`

### 16. Imports ⚠️ ([part2_house_value_regression.py:5](Neural_Networks_fz823/part2_house_value_regression.py:5))
Added three lines. The original five imports are unchanged.
```python
from torch import nn                                   # layers and loss (nn.Linear, nn.MSELoss)
from sklearn.metrics import root_mean_squared_error    # used by evaluate()
from sklearn.model_selection import train_test_split   # used by main()
```
All of these come from packages already in `requirements.txt`, so nothing new needs to be installed on LabTS.

### 17. `Regressor.__init__` signature ⚠️ ([part2_house_value_regression.py:14](Neural_Networks_fz823/part2_house_value_regression.py:14))
```python
# before
def __init__(self, x, nb_epoch = 1000):
# after
def __init__(self, x, nb_epoch = 150, hidden_layers = (64, 64),
             learning_rate = 1e-3, batch_size = 64):
```
- **`nb_epoch` default changed from 1000 to 150.** Validation RMSE plateaus by about 150 epochs, and 1000 would make any LabTS test that uses the defaults take about a minute. Passing `nb_epoch=...` explicitly still works exactly as before.
- **Three new optional arguments.** Every one has a default, so `Regressor(x)` and `Regressor(x, nb_epoch=10)` still work. That is the spec's rule for added arguments.
- A 2-line comment above the `def` and 3 docstring lines describing the new arguments were also added.

### 18. `Regressor.__init__` body ✅ ([part2_house_value_regression.py:34](Neural_Networks_fz823/part2_house_value_regression.py:34))
```python
# before
# Replace this code with your own
X, _ = self._preprocessor(x, training = True)
self.input_size = X.shape[1]
self.output_size = 1
self.nb_epoch = nb_epoch
return
# after  (first 4 lines kept; the "Replace this code" comment removed)
X, _ = self._preprocessor(x, training = True)
self.input_size = X.shape[1]
self.output_size = 1
self.nb_epoch = nb_epoch
self.learning_rate = learning_rate
self.batch_size = batch_size

layers = []
n_in = self.input_size
for n_out in hidden_layers:
    layers += [nn.Linear(n_in, n_out), nn.ReLU()]
    n_in = n_out
layers.append(nn.Linear(n_in, self.output_size))
self.model = nn.Sequential(*layers)
return
```

### 19. `Regressor.fit` ✅ ([part2_house_value_regression.py:165](Neural_Networks_fz823/part2_house_value_regression.py:165))
```python
# before
X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
return self
# after  (the template's two lines kept, training loop added between them)
X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
optimiser = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
loss_fn = nn.MSELoss()
self.model.train()

for _ in range(self.nb_epoch):
    permutation = torch.randperm(len(X))
    for i in range(0, len(X), self.batch_size):
        batch = permutation[i:i + self.batch_size]
        optimiser.zero_grad()
        loss = loss_fn(self.model(X[batch]), Y[batch])
        loss.backward()
        optimiser.step()
return self
```

### 20. `Regressor.predict` ✅ ([part2_house_value_regression.py:207](Neural_Networks_fz823/part2_house_value_regression.py:207))
```python
# before
X, _ = self._preprocessor(x, training = False) # Do not forget
pass
# after
X, _ = self._preprocessor(x, training = False) # Do not forget
self.model.eval()
with torch.no_grad():
    return self._revert_preprocessor(self.model(X))
```

### 21. `Regressor.evaluate` ✅ ([part2_house_value_regression.py:237](Neural_Networks_fz823/part2_house_value_regression.py:237))
```python
# before
return 0 # Replace this code with your own
# after
return float(root_mean_squared_error(y, self.predict(x)))
```

### 22. `main()` ⚠️ ([part2_house_value_regression.py:269](Neural_Networks_fz823/part2_house_value_regression.py:269))
The template says *"This function is not called by LabTS, so feel free to change it."*
```python
# before
x_train = data.loc[:, data.columns != output_label]
y_train = data.loc[:, [output_label]]
regressor = Regressor(x_train, nb_epoch = 10)
regressor.fit(x_train, y_train)
save_regressor(regressor)
error = regressor.evaluate(x_train, y_train)
print(f"\nRegressor error: {error}\n")
# after
x = data.loc[:, data.columns != output_label]
y = data.loc[:, [output_label]]
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size = 0.2, random_state = 42)
torch.manual_seed(0)
regressor = Regressor(x_train)
regressor.fit(x_train, y_train)
save_regressor(regressor)
print(f"\nTrain RMSE: {regressor.evaluate(x_train, y_train)}")
print(f"Validation RMSE: {regressor.evaluate(x_val, y_val)}\n")
```
The `data = pd.read_csv("housing.csv")` line and the template's comments above it are unchanged.

### 23. `if __name__ == "__main__":` ⚠️ ([part2_house_value_regression.py:301](Neural_Networks_fz823/part2_house_value_regression.py:301))
```python
# before
main()
# after
import part2_house_value_regression
part2_house_value_regression.main()
```
Without this, running the file directly would save the pickle with its class recorded as `__main__.Regressor`, and LabTS (which imports the module) might not find that class.
With it, the class is recorded as `part2_house_value_regression.Regressor`.

---

## Files added

| File | Where | Submitted? |
|---|---|---|
| `part2_model.pickle` | repo | **Yes.** It's a required deliverable and still needs `git add` |
| `.venv/` | `Coursework_1/` (outside repo) | No. Python environment pinned to `requirements.txt` |
| `check_part1.py` | `Coursework_1/` | No. Gradient check and preprocessor self-test |
| `IMPLEMENTATION_NOTES.md` | `Coursework_1/` | No. Explanations and derivations |
| `CHANGES.md` | `Coursework_1/` | No. This file |
