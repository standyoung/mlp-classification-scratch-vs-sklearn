# 파이썬 라이브러리인 scikit-learn을 이용해 iris classification하기
from sklearn.datasets import load_iris
import urllib.request  # iris data 불러오기 by urlib.request()
import numpy as np  # 데이터 타입이 numpy.ndarray이기 때문에 numpy 라이브러리를 통해 연산하면 편함
from sklearn.model_selection import train_test_split
# scikitlearn을 사용하여 dataset를 train set와 test set로 분할
# train set는 모델을 훈련하는 데 사용하고 test set는 최종 훈련된 모델의 성능을 평가하는 데 사용함
from sklearn.preprocessing import LabelEncoder  # MLP 모델 학습을 위해 데이터 정수로 처리
from sklearn.preprocessing import OneHotEncoder  # 1,0으로 카테고리 변경

# 1. iris data set 불러오기

def load_iris_dataset():
    # uci 사이트에서 제공하는 iris 데이터 세트가 있는 url
    iris_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # urlib.request()로 데이터 가져오기
    urlib = urllib.request.urlopen(iris_data_url)
    data = urlib.read().decode("utf-8")
    data = data.split("\n")

    # 데이터 리스트를 DataFrame으로 변환
    data = [row.split(",") for row in data if row]  # 각 행을 콤마로 나누어 리스트로 변환
    columns = ["sepal_length", "sepal_width",
               "petal_length", "petal_width", "class"]  # 열 이름 지정
    df = pd.DataFrame(data, columns=columns)  # DataFrame 생성

    # class int형과 feature는 float형으로 변경
    df.iloc[:, 0:4] = df.iloc[:, 0:4].astype(float)
    df['class'] = df['class'].map(
        {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

    # NumPy의 np.zeros()는 주어진 형태와 타입을 갖는 0을 갖는 어레이를 반환
    X = df.iloc[:, 0:4].values
    y = df.iloc[:, 4].values

    return X, y

# X, y 데이터 분리
X, y = load_iris_dataset()

# 데이터셋을 랜덤하게 나누는 함수 정의
def train_test_split_sc(X, y, test_size=0.3, random_state=None):
    np.random.seed(random_state)  # 시드 설정

    # 데이터 개수 계산
    num_samples = X.shape[0]

    # 데이터를 무작위로 섞기 위해 인덱스를 생성하고 셔플
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    # 테스트 데이터의 크기 계산
    num_test_samples = int(num_samples * test_size)

    # 테스트와 학습 데이터셋 분할
    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

# train set, test set 분리
X_train, X_test, y_train, y_test = train_test_split_sc(X, y)

# 3. Hyperparameter 설정
learning_rate = 0.01  # 초기 학습률
epochN = 100  # 에폭 수 설정
hidden_nodes = [150, 50, 25]  # 은닉층 크기

# Feature와 label 개수 정의
num_features = X_train.shape[1]  # 4
num_labels = len(np.unique(y_train))  # 3

# 데이터셋 준비 (예시로 iris 데이터 사용)

data = load_iris()
X, y = data.data, data.target

# one-hot 인코딩
num_labels = len(np.unique(y))
y_one_hot = np.eye(num_labels)[y]

# 4. MLP 모델

# 활성화 함수 정의
# ReLU 함수
def relu(Z):
    return np.maximum(0, Z)

# ReLU 함수의 미분
def relu_derivative(Z):
    return (Z > 0).astype(float)

# 소프트맥스 함수 (출력층에 사용)
def softmax(Z):
    Z = np.array(Z, dtype=np.float64)  # Z를 float64로 변환하여 안정성 확보
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# 순전파 (Forward Propagation)
def forward_propagation(X):
    Z1 = np.dot(X, w1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, w2) + b2
    A2 = relu(Z2)

    Z3 = np.dot(A2, w3) + b3
    A3 = relu(Z3)

    Z_out = np.dot(A3, w_out) + b_out
    A_out = softmax(Z_out)

    cache = (Z1, A1, Z2, A2, Z3, A3, Z_out, A_out)
    return A_out, cache

# 역전파 (Backward Propagation)
def backward_propagation(X, y, cache):
    Z1, A1, Z2, A2, Z3, A3, Z_out, A_out = cache
    m = X.shape[0]

    # 출력층의 오차
    dZ_out = A_out - y
    dw_out = np.dot(A3.T, dZ_out) / m
    db_out = np.sum(dZ_out, axis=0, keepdims=True) / m

    # 은닉층 3의 오차
    dA3 = np.dot(dZ_out, w_out.T)
    dZ3 = dA3 * relu_derivative(Z3)
    dw3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    # 은닉층 2의 오차
    dA2 = np.dot(dZ3, w3.T)
    dZ2 = dA2 * relu_derivative(Z2)
    dw2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # 은닉층 1의 오차
    dA1 = np.dot(dZ2, w2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dw1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    grads = (dw1, db1, dw2, db2, dw3, db3, dw_out, db_out)
    return grads

# 가중치 업데이트 함수
def update_parameters(grads, learning_rate):
    global w1, b1, w2, b2, w3, b3, w_out, b_out
    dw1, db1, dw2, db2, dw3, db3, dw_out, db_out = grads

    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    w3 -= learning_rate * dw3
    b3 -= learning_rate * db3
    w_out -= learning_rate * dw_out
    b_out -= learning_rate * db_out

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y_one_hot, test_size=0.3, random_state=1234)

# Feature와 label 개수 정의
num_features = X_train.shape[1]

# 가중치 및 편향 초기화 (입력층, 은닉층 및 출력층)
w1 = np.random.randn(
    num_features, hidden_nodes[0]) * np.sqrt(2. / num_features)
w2 = np.random.randn(
    hidden_nodes[0], hidden_nodes[1]) * np.sqrt(2. / hidden_nodes[0])
w3 = np.random.randn(
    hidden_nodes[1], hidden_nodes[2]) * np.sqrt(2. / hidden_nodes[1])
w_out = np.random.randn(
    hidden_nodes[2], num_labels) * np.sqrt(2. / hidden_nodes[2])

b1 = np.zeros((1, hidden_nodes[0]))
b2 = np.zeros((1, hidden_nodes[1]))
b3 = np.zeros((1, hidden_nodes[2]))
b_out = np.zeros((1, num_labels))

# 모델 학습
for epoch in range(epochs):
    # 순전파
    A_out, cache = forward_propagation(X_train)

    # 손실 계산 (크로스 엔트로피 손실)
    loss = -np.mean(np.sum(y_train * np.log(A_out + 1e-9), axis=1))
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    # 역전파
    grads = backward_propagation(X_train, y_train, cache)

    # 가중치 업데이트
    update_parameters(grads, learning_rate)

# 모델 평가
A_out_test, _ = forward_propagation(X_test)
y_pred_test = np.argmax(A_out_test, axis=1)
y_true_test = np.argmax(y_test, axis=1)

test_accuracy = np.mean(y_pred_test == y_true_test) * 100
print(f"Test Accuracy: {test_accuracy:.2f}%")
