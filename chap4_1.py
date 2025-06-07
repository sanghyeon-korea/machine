import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# sigmoid 함수
def sigmoid(z):
    p = 1/(1+np.exp(-z))
    return p
'''
1. onehot_encoding
1. 입력 -> y
2. class 몇개? -> np.unique(array) 사용
3. class 개수 만큼 y0~ yQ 를 만들어줘야함
3. for문 사용, ex) if y = a y0, y1, y2 = 1, 0, 0
for 보다 더 나은 방법 사용. 

'''
#원핫 인코딩 함수
def onehot_encoding(y) :
    Q = np.unique(y) #클래스 수
    N = len(y) #데이터 수
    onehot_y = np.zeros((N, len(Q)), dtype=int)
    for i in range(N) :
        Q_index = np.where(Q == y[i])[0][0] #Q = y[i]를 찾아서
        print(Q_index)
        onehot_y[i, Q_index] = 1 #그곳에 1을 넣음
    return onehot_y


'''
2. Two_layer_neural_network
 1.입력받은 데이터에서 input 속성 수(특징 개수), output class 수 자동으로 체크.
  -> output class는 unique를 사용하거나 아니면 onehot_encoding을 응용하면 될 듯
 2.hidden layer node 수 설정 -> 입력받는 게 히든 레이어 수 결론적으로 데이터와 히든레이어 수를 입력

 3.인풋 속성 수, 아웃풋 클래스 수, 히든 노드수를 사용해서 웨이트 메트릭스 생성. 
  -> 아마 랜덤을 걸어서 하면 될 듯. 랜덤의 개수는 v -> L(입력받은 노드수) by M+1(특징수 + 1)
 4.차라리 x를 M+1 by N으로 하려면 특징수 체크는 행의 개수 -1을 하면 될 듯?

 5.도출해야할 것 -> y_hat....  Q by N으로 나오는 꼴일 거임. 강의자료 확인

'''
def Two_layer_neural_network(x, y , L) :
    y_class = np.unique(y)
    Q = len(y_class) #클래스 수
    M = len(x) - 1 #특징 수
    v = np.random.randn(L, M+1)   #이부분 1을 넣으면 y_hat이 양수 나옴. 범위 바꿔줘야 음수도 나옴.
    w = np.random.randn(Q, L+1)   #random.rand 를 뒤에 수를 빼지 않는 이상 양수가 나오므로 randn를 사용
    # v = np.random.rand(L, M+1)   #이부분 1을 넣으면 y_hat이 양수만 나옴. 범위 바꿔줘야 음수도 나옴.
    # w = np.random.rand(Q, L+1)
    alpha = v@x #
    b = sigmoid(alpha)
    bias_row = np.ones((1, b.shape[1]))
    b = np.vstack([b, bias_row])
    beta = w@b
    y_hat = sigmoid(beta)
    print("y_hat의 최소값 =", np.min(y_hat))
    
    return y_hat

'''
3. Accuracy 함수
0.5 기준 0~1 변환
max 기준 0~1 변환 두가지 다 할 것
전체 데이터에서 맞은 개수를 하면 정확도 나올 듯
'''
def Accuracy_05(y, y_hat) : #임계값 0.5 기준 정확도 체크
    y_hat = y_hat.T
    N = len(y_hat)
    Q = len(y_hat[0])
    accuracy = 0
    y_hat_test = np.zeros((N, Q), dtype=int)
    count = 0
    print(N)
    print(Q)
    
    for i in range(Q) : #0.5 이상 1 아닌 경우 0
        for j in range(N) :
            if y_hat[j, i] >= 0.5 :
                y_hat_test[j,i] = 1
            else:
                y_hat_test[j,i] = 0
                
    for i in range(N) : #y_hat = y 카운트
        if np.all(y_hat_test[i, :] == y[i, :]):
            count += 1
    
    accuracy = count / N #정확도
    
    return y_hat_test, accuracy



def Accuracy_max(y, y_hat) : #한 데이터에 대한 y_hat중 가장 큰 것을 1로 설정하여 정확도 체크
    y_hat = y_hat.T
    N = len(y_hat)
    Q = len(y_hat[0])
    accuracy = 0
    y_hat_test = np.zeros((N, Q), dtype=int)
    count = 0
    print(N)
    print(Q)
    #가장 큰 y_hat을 1로 설정
    for i in range(N) :
        max_j = np.argmax(y_hat[i, :])    
        y_hat_test[i, max_j] = 1
    #y_hat = y 카운트
    for i in range(N) :
        if np.all(y_hat_test[i, :] == y[i, :]):
            count += 1
    
    accuracy = count / N #정확도
    
    return y_hat_test, accuracy

#이번 데이터는 특징수 3개 데이터 수 1800개
# M = 3, N = 1800, alpha = L(히든레이어 개수) by 1800, x = 4 by 1800 트랜스 포즈 해야함(더미 포함)
#v = L by 4, beta = 6 by N y_hat = 6 by N, w = 6 by L + 1

df = pd.read_csv('C:\\Users\\sanghyeon\\Desktop\\2021146019AHN\\3-1\\machine\\NN_data.csv')
df.insert(3, 'x3', 1)
dfnp = df.to_numpy()
y = dfnp[:, 4]
x = dfnp[:, 0:4]
x = x.T
onehot_y = onehot_encoding(y)
y_hat = Two_layer_neural_network(x, y, 3)
y__, accuracy_by_05 = Accuracy_05(onehot_y, y_hat)
y_ac_max, accuracy_by_max = Accuracy_max(onehot_y, y_hat)






