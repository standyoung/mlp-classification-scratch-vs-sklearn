import numpy as np # 데이터 타입이 numpy.ndarray이기 때문에 numpy 라이브러리를 통해 연산하면 편함
from sklearn.datasets import fetch_openml # MNIST data set 불러오기 by fetch_openml()
from sklearn.metrics import accuracy_score

# 1. MNIST data set 불러오기
mnist = fetch_openml(name='mnist_784', as_frame=False) 
# MNIST data를 numpy.ndarray 형태로 불러오기
# as_frame = True일 경우 DataFrame으로 변환되므로 False를 써줘야 함
X, y = mnist["data"], mnist["target"] # 특징 벡터를 X에 넣고 label 벡터를 y에 넣음
# 7만개의 이미지 28x28 픽셀로 784개의 특징이 있음

# 2. 데이터 전처리
X = X / 255 # 0~255까지의 픽셀 이미지 데이터를 정규화 시킴
digits = 10 # 0~9
examples = y.shape[0]
y = y.reshape(1, examples)
Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples) # one-hot encode labels
    
m = 60000
m_test = X.shape[0] - m
# sklearn으로 불러온 MNIST는 7만개 이미지 중 6만개는 학습용, 1만개는 테스트용으로 정렬되어있음
# 6만개의 학습용 이미지
# 나머지 1만개의 테스트용 이미지

X_train, X_test = X[:m].T, X[m:].T
Y_train, Y_test = Y_new[:,:m], Y_new[:,m:]
# 데이터를 전치시켜 train set, test set에 넣음
# dataset를 train set와 test set로 분할하기
   
shuffle_index = np.random.permutation(m) 
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]
# 학습을 위해 train set를 섞음

# 3. parameter 조정
n_x = X_train.shape[0]
n_h = 64
learning_rate = 4 # 학습속도
beta = .9
batch_size = 128 # 배치 사이즈 128로 설정
batches = -(-m // batch_size)
np.random.seed(138)

# initialization
params = { "W1": np.random.randn(n_h, n_x) * np.sqrt(1. / n_x),
          "b1": np.zeros((n_h, 1)) * np.sqrt(1. / n_x),
          "W2": np.random.randn(digits, n_h) * np.sqrt(1. / n_h),
          "b2": np.zeros((digits, 1)) * np.sqrt(1. / n_h) }

V_dW1 = np.zeros(params["W1"].shape)
V_db1 = np.zeros(params["b1"].shape)
V_dW2 = np.zeros(params["W2"].shape)
V_db2 = np.zeros(params["b2"].shape)

# 활성함수 : 로지스틱 시그모이드 함수 사용
def sigmoid_activation(z):
    s = 1. / (1. + np.exp(-z)) # f(x) = 1 / (1 + exp(-x))
    return s

def compute_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1./m) * L_sum
    return L

def feed_forward(X, params):
    cache = {}
    cache["Z1"] = np.matmul(params["W1"], X) + params["b1"]
    cache["A1"] = sigmoid_activation(cache["Z1"])
    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]
    cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)
    return cache

def back_propagate(X, Y, params, cache):
    dZ2 = cache["A2"] - Y
    dW2 = (1./m_batch) * np.matmul(dZ2, cache["A1"].T)
    db2 = (1./m_batch) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.matmul(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid_activation(cache["Z1"]) * (1 - sigmoid_activation(cache["Z1"]))
    dW1 = (1./m_batch) * np.matmul(dZ1, X.T)
    db1 = (1./m_batch) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads


# train
for i in range(10):

    permutation = np.random.permutation(X_train.shape[1])
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = Y_train[:, permutation]

    for j in range(batches):

        begin = j * batch_size
        end = min(begin + batch_size, X_train.shape[1] - 1)
        
        # 처음부터 끝까지
        X = X_train_shuffled[:, begin:end]
        Y = Y_train_shuffled[:, begin:end]
        m_batch = end - begin

        cache = feed_forward(X, params)
        grads = back_propagate(X, Y, params, cache)

        V_dW1 = (beta * V_dW1 + (1. - beta) * grads["dW1"])
        V_db1 = (beta * V_db1 + (1. - beta) * grads["db1"])
        V_dW2 = (beta * V_dW2 + (1. - beta) * grads["dW2"])
        V_db2 = (beta * V_db2 + (1. - beta) * grads["db2"])

        params["W1"] = params["W1"] - learning_rate * V_dW1
        params["b1"] = params["b1"] - learning_rate * V_db1
        params["W2"] = params["W2"] - learning_rate * V_dW2
        params["b2"] = params["b2"] - learning_rate * V_db2

    cache = feed_forward(X_train, params)
    train_cost = compute_loss(Y_train, cache["A2"])
    cache = feed_forward(X_test, params)
    test_cost = compute_loss(Y_test, cache["A2"])

cache = feed_forward(X_test, params)
predictions = np.argmax(cache["A2"], axis=0)
labels = np.argmax(Y_test, axis=0)
accuracy = accuracy_score(predictions, labels)

# 정확도(정확히 맞힌 개수/전체 데이터 수) 측정
# score() 훈련된 모델의 성능을 측정, 매개변수인 특징과 정답 데이터를 전달함
# 올바르게 예측한 개수의 비율을 반환함
print("MNIST MLP from scratch 테스트 정확도 : {:.2f}".format(accuracy*100))
