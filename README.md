# Automatic Gradient Descent (Scalar)

Implements backpropagation (reverse mode autodiff) over a dynamically constructed Directed Acyclic Graph (DAG) and a basic neural network library similar to PyTorch. The DAG works only with scalar values, meaning each neuron is broken down into smaller operations like additions and multiplications. Despite this, it's capable of building deep neural networks for tasks like binary classification, as demonstrated in the notebook example.

## Example usage
```python
x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
n(x)
```

```python
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.6],
    [6.0, 1.0, 5.0],
    [1.0, 3.0, 2.0]
]
ys = [1.0, -1.0, 1.0, -1.0]
```

```python
for k in range(20):
    #forward pass
    ypred = [n(x) for x in xs]
    loss = sum([(ygt - yout) ** 2 for ygt, yout in zip(ys, ypred)])

    # zero grad
    for p in n.parameters():
        p.grad = 0
    
    #backward pass
    loss.backward()

    #update
    for p in n.parameters():
        p.data += -0.05 * p.grad
    
    print(k, loss.data)
```
