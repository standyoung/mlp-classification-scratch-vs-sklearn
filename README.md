## 🧠 Multi-Layer Perceptron from Scratch
This project aims to implement a Multi-Layer Perceptron (MLP) model from scratch using only Python and NumPy to solve a multi-class classification problem. The performance of this custom implementation will then be compared against scikit-learn's MLPClassifier.

### 📌 Goals
- Implement Feed-forward and Back-propagation from scratch.
- Apply Stochastic Gradient Descent (SGD) optimization.
- Implement the ReLU activation function, Softmax output layer, and Cross-Entropy loss function.
- Compare performance: Custom-implemented fully connected MLP vs. Scikit-learn's MLPClassifier.

### 📈 Performance Improvement
- Initial Accuracy: 77%
- After introducing He initialization: 93.3%
- Achieved accuracy levels comparable to scikit-learn's MLPClassifier.

### 📁 Project Structure
```
.
├── iris_scikitlearn.py       # Iris Dataset - scikit-learn MLPClassifier
├── iris_scratch.py           # Iris Dataset - scratch MLP implementation
├── mnist_scikitlearn.py      # MNIST Dataset - scikit-learn MLPClassifier
└── mnist_scratch.py          # MNIST Dataset - scratch MLP implementation
```
