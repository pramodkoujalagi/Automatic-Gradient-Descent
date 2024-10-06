# Automatic Gradient Descent (Scalar)

Implements backpropagation (reverse mode autodiff) over a dynamically constructed Directed Acyclic Graph (DAG) and a basic neural network library similar to PyTorch. The DAG works only with scalar values, meaning each neuron is broken down into smaller operations like additions and multiplications. Despite this, it's capable of building deep neural networks for tasks like binary classification, as demonstrated in the notebook example.
