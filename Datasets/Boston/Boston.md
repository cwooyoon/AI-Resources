# Boston dataset

출처: 모두의딥러닝(조태호 저)

## 보스턴 집값 예측

1978년, 집값에 가장 큰 영향을 미치는 것이 ‘깨끗한 공기’라는 연구 결과가 하버드대학교 도시개발학과에서 발표됨

이들은 자신의 주장을 뒷받침하기 위해 집값의 변동에 영향을 미치는 여러 가지 요인을 모아서 환경과 집값의 변동을 보여주는 데이터셋을 만듦

그로부터 수십 년 후, 이 데이터셋은 머신러닝의 선형 회귀를 테스트하는 가장 유명한 데이터로 쓰이고 있음

<img src="https://user-images.githubusercontent.com/54765256/90978914-2492bd00-e58c-11ea-8585-60472f27dc33.png">

Index가 506개이므로 총 샘플의 수는 506개이고, 컬럼 수는 14개이므로 13개의 속성과 1개의 클래스로 이루어졌음을 짐작할 수 있음

<img src="https://user-images.githubusercontent.com/54765256/90978938-4ab85d00-e58c-11ea-9ea9-c2dc2538a2cf.png">

특히 마지막 컬럼을 보면 지금까지와는 다름
클래스로 구분된 게 아니라 가격이 나와 있음

<img src="https://user-images.githubusercontent.com/54765256/90978948-61f74a80-e58c-11ea-9d45-1cc9c75b122c.png">

선형 회귀 데이터는 마지막에 참과 거짓을 구분할 필요가 없음
출력층에 활성화 함수를 지정할 필요도 없음

모델의 학습이 어느 정도 되었는지 확인하기 위해 예측 값과 실제 값을 비교하는 부분을 추가함

```
#-*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy
import pandas as pd
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

df = pd.read_csv("../dataset/housing.csv", delim_whitespace=True, header=None)
'''
print(df.info())
print(df.head())
'''
dataset = df.values
X = dataset[:,0:13]
Y = dataset[:,13]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error',
              optimizer='adam')

model.fit(X_train, Y_train, epochs=200, batch_size=10)

# 예측 값과 실제 값의 비교
Y_prediction = model.predict(X_test).flatten()
for i in range(10):
    label = Y_test[i]
    prediction = Y_prediction[i]
    print("실제가격: {:.3f}, 예상가격: {:.3f}".format(label, prediction))
```

flatten( ) 함수 :

   데이터 배열이 몇 차원이든 모두 1차원으로 바꿔 읽기 쉽게 해 주는 함수
   
range('숫자')는 0부터 ‘숫자-1’만큼 차례대로 증가하며 반복되는 값을 만듦

즉, range(10)은 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]를 말함

for in 구문을 사용해서 10번 반복하게 함
















