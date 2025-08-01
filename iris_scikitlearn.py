from sklearn.datasets import load_iris  # iris data 불러오기 by load_iris()
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
# scikitlearn을 사용하여 dataset를 train set와 test set로 분할
# train set는 모델을 훈련하는 데 사용하고 test set는 최종 훈련된 모델의 성능을 평가하는 데 사용함
from sklearn.metrics import accuracy_score  # 예측값과 정답 배열을 넣어 정확도 평가
from tqdm import tqdm  # tqdm 불러오기


# 1. iris data set 불러오기
iris = load_iris()
X, y = iris.data, iris.target
# data는 특징인 꽃받침의 길이,너비, 꽃잎의 길이,너비가 있음
# target은 Setosa를 0, Versicolor를 1, Virginica 2로 저장되어 있음

# train set, test set 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234)  # train set 70%, test set 30%로 분할

# MLP 모델 생성
mlp = MLPClassifier(hidden_layer_sizes=(150, 50, 25),
                    activation='relu', solver='sgd', learning_rate_init=0.001,)
# 은닉층 3개, 1번째:노드150개 2번째:50개, 3번째:25개
# 활성화 함수 : ReLU
# solver : 최적화 알고리즘 - SGD
# init learning rate : 0.001로 설정
# epoch의 수 : 100번 반복, parameter 업데이트

# MLP 모델 학습
epochs = 100
losses = []

# MLP 모델 학습
for epoch in tqdm(range(1, epochs + 1)):
    mlp.fit(X_train, y_train)
    losses.append(mlp.loss_)
    # 매 10 Epoch마다 출력
    if epoch % 10 == 0:
        print(f"\nEpoch {epoch}/{epochs}, Loss: {mlp.loss_:.4f}")

# MLP 모델 예측
y_pred = mlp.predict(X_test)

# 정확도 측정
accuracy = accuracy_score(y_test, y_pred)

# 정확도 평가결과
print("Iris MLP by scikitlearn 테스트 정확도 : {:.2f}%".format(accuracy*100))
