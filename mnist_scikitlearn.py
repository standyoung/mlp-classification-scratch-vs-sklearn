# 파이썬 라이브러리인 scikit-learn을 이용해 mnist classification하기
from sklearn.neural_network import MLPClassifier #
from sklearn.datasets import fetch_openml # MNIST data set 불러오기 by fetch_openml()
from sklearn.model_selection import train_test_split 
# scikitlearn을 사용하여 dataset를 train set와 test set로 분할
# train set는 모델을 훈련하는 데 사용하고 test set는 최종 훈련된 모델의 성능을 평가하는 데 사용함
from sklearn.metrics import accuracy_score # 예측값과 정답 배열을 넣어 정확도 평가

# 1. MNIST data set 불러오기
mnist = fetch_openml('mnist_784', as_frame=False) # as_frame = True일 경우 DataFrame으로 변환되므로 False를 써줘야 함
X, y = mnist["data"], mnist["target"] # X에 특징 데이터 넣기 784개, y에 목표값 넣기 7만개

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # train set 70%, test set 30%로 분할

# 2. MLP 모델 생성 & parameter 조정
mlp = MLPClassifier(hidden_layer_sizes=(784, 100, 25), activation='logistic', max_iter=1000, random_state=138)
    # 은닉층 3개, 1번째:노드784개 2번째:100개, 3번째:25개
    # 활성화 함수 : 로지스틱 시그모이드
    # epoch의 수 : 1000번
    # random state : 138

# 3. MLP 모델 학습
mlp.fit(X_train, y_train)
    # fit() scikitlearn 모델을 훈련함, 매개변수로 훈련해 사용할 특징과 정답 data를 전달함

# 4. MLP 모델 예측
y_pred = mlp.predict(X_test)
    # predict() scikitlearn 모델을 훈련하고 예측함, 특징 데이터만 매개변수로 받음

# 5. 정확도 평가결과
accuracy = accuracy_score(y_test, y_pred)
print("MNIST MLP by scikitlearn 테스트 정확도 : {:.2f}".format(accuracy*100))
# 정확도(정확히 맞힌 개수/전체 데이터 수) 측정
# score() 훈련된 모델의 성능을 측정, 매개변수인 특징과 정답 데이터를 전달함
# 올바르게 예측한 개수의 비율을 반환함