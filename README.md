## 🧠 Multi-Layer Perceptron from Scratch
이 프로젝트는 다중 클래스 분류 문제를 해결하기 위한 **Multi-Layer Perceptron (MLP)** 모델을 **Python과 NumPy만을 이용해 처음부터 직접 구현**하고, **scikit-learn의 MLPClassifier**와 성능을 비교하는 것을 목표로 합니다.

### 📌 목표
- Feed-forward 및 Back-propagation 직접 구현
- Stochastic Gradient Descent (SGD) 기반 최적화 적용
- ReLU 활성화 함수, Softmax 출력층, Cross-Entropy 손실 함수 구현
- 성능 비교: 직접 구현한 fully connected MLP vs. Scikit-learn MLPClassifier
- 실험 데이터셋:
  - Iris 데이터셋
  - MNIST 손글씨 숫자 데이터셋

### 📈 성능 개선
- 초기 정확도: 77%
- He 초기화 기법 도입 후: 93.3%
- scikit-learn MLPClassifier와 동등한 정확도 수준 달성

### 📁 프로젝트 구조
```
.
├── iris_scikitlearn.py       # Iris 데이터셋 - scikit-learn MLPClassifier
├── iris_scratch.py           # Iris 데이터셋 - scratch MLP 구현
├── mnist_scikitlearn.py      # MNIST 데이터셋 - scikit-learn MLPClassifier
└── mnist_scratch.py          # MNIST 데이터셋 - scratch MLP 구현
```
