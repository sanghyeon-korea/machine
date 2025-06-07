import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


# sigmoid 함수
def sigmoid(z):
    p = 1/(1+np.exp(-z))
    return p

#원핫 인코딩 함수
def onehot_encoding(y) :
    Q = np.unique(y) #클래스 수
    N = len(y) #데이터 수
    onehot_y = np.zeros((N, len(Q)), dtype=int)
    for i in range(N) :
        Q_index = np.where(Q == y[i])[0][0] #Q = y[i]를 찾아서
        onehot_y[i, Q_index] = 1 #그곳에 1을 넣음
    return onehot_y, len(Q)

#set_w, v
def set_w_v(M, L, Q):
    v = np.random.randn(L, M+1)
    w = np.random.randn(Q, L+1)
    return v, w

#forward propagation
def forward_propagation(x_n, v, w) :
    alpha = v@x_n
    b = sigmoid(alpha)
    b = np.vstack([b, [[1]]])
    beta = w@b
    y_hat = sigmoid(beta) #Qx1
    return y_hat ,b

#데이터 쪼개는 함수
def div_data(data, a, b, c):
    np.random.shuffle(data) #셔플을 통해 데이터를 섞음
    first_div = int(len(data) * a / 10) #training set과 validation set을 나누기 위한 기준
    second_div = int(len(data) - len(data) * c / 10) #validation set과 test set을 나누기 위한 기준
    training_set = data[0:first_div, :]
    validation_set = data[first_div:second_div, :]
    test_set = data[second_div:, :]
    return training_set, validation_set, test_set

def Accuracy_max(y, y_hat):  # y_hat: (N, Q)
    # 1) 전치 제거 — y_hat은 (N, Q) 그대로
    N, Q = y_hat.shape

    # 2) 예측을 one-hot으로 변환
    y_hat_test = np.zeros((N, Q), dtype=int)
    max_j = np.argmax(y_hat, axis=1)        # 각 행(샘플)마다 최댓값 인덱스
    y_hat_test[np.arange(N), max_j] = 1

    # 3) 실제 y가 1차원 레이블 형태면 one-hot으로 변환
    if y.ndim == 1:
        onehot_y = np.zeros((N, Q), dtype=int)
        for i in range(N):
            onehot_y[i, int(y[i])] = 1
        y = onehot_y

    # 4) 정확도 계산
    count = np.sum(np.all(y_hat_test == y, axis=1))
    accuracy = count / N

    return y_hat_test, accuracy




def back_propagation(x_n, y_n, v, w, learnig_rate):
    """
    x_n: (M+1 x 1)
    y_n: (Q x 1) - one-hot
    v: L x (M+1)
    w: Q x (L+1)
    """

    # 순전파
    y_hat, b = forward_propagation(x_n, v, w)

    #(출력층)
    sigma = 2 * (y_hat - y_n) * y_hat * (1 - y_hat)  # Q x 1

    # ----- v 먼저 업데이트 -----
    temp = (w.T @ sigma)[:-1]  # (L+1 x Q) @ (Q x 1) = (L+1 x 1), 마지막 바이어스 제외
    db = b[:-1] * (1 - b[:-1])  # (L x 1)
    grad_v = temp * db  # (L x 1)
    v -= learnig_rate * (grad_v @ x_n.T)  # L x (M+1)

    # ----- 그 다음 w 업데이트 -----
    w -= learnig_rate * (sigma @ b.T)  # Q x (L+1)

    return v, w

#전체 배치에 대한 제곱 오차의 평균
def compute_MSE(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


#training 함수
def training_with_metrics(x_train, y_train, v, w, learning_rate, epoch=100):
    N = x_train.shape[1]
    acc_list = []
    mse_list = []

    for e in range(epoch):
        idx = np.random.permutation(N)

        for i in idx:
            x_n = x_train[:, i].reshape(-1, 1)
            y_n = y_train[i].reshape(-1, 1)
            v, w = back_propagation(x_n, y_n, v, w, learning_rate)

        #  Epoch 끝난 후 전체 Training Set에 대해 Accuracy, MSE 계산 
        y_hat_list = []
        for i in range(N):
            y_hat, _ = forward_propagation(x_train[:, i].reshape(-1, 1), v, w)
            y_hat_list.append(y_hat.T)  # (1, Q)

        y_hat_all = np.vstack(y_hat_list)  # (N, Q)
        _, acc = Accuracy_max(y_train, y_hat_all)
        mse = compute_MSE(y_train, y_hat_all)

        acc_list.append(acc)
        mse_list.append(mse)
    
    return v, w, acc_list, mse_list






    

#이번 데이터는 특징수 3개 데이터 수 1800개
# M = 3, N = 1800, alpha = L(히든레이어 개수) by 1800, x = 4 by 1800 트랜스 포즈 해야함(더미 포함)
#v = L by 4, beta = 6 by N y_hat = 6 by N, w = 6 by L + 1

'''웨이트 매트릭스
mse를 wlq로 편미분 -> 2(y_hat_qn-y_qn)*y_hat_qn(1-y_hat_qn)b_ln
y_hat -> Qx1, y -> Qx1, b_ln -> L+1x1 구하려는 w 는 qxL+1 따라서 b_ln을 전치할 것

mse를 bln으로 편미분 -> q=0~q=Q-1까지 시그마_qn * wlq, bln(1-b_ln)Xmn q를 날리면 나머지 해서 v= LxM+1이 나옴

n개의 훈련데이터는 항상 epoch 보기 전 계속 shuffle
'''

'''
for 부분 -> 순전파해서 y_hat 구하고 역전파 해서 오차로부터 웨이트 업데이트
주의 -> v부터 업데이트 할 것 이유는 w는 학습된 w가 아닌 초기화된 w를 써야하는데 w부터 업데이트하면
학습된 w를 사용할 가능성이 높아진다
'''


if __name__ == "__main__" : #메인 함수
    # 1. 데이터 가져오기, 더미 추가
    df = pd.read_csv('C:\\Users\\sanghyeon\\Desktop\\2021146019AHN\\3-1\\machine\\NN_data.csv')
    df.insert(3, 'x3', 1)
    dfnp = df.to_numpy()

    # 2. 데이터 분할 (7:3)
    training_set, _, test_set = div_data(dfnp, 7, 0, 3)  # b 파라미터 사용 안 함

    # 3. 분할된 x, y 처리
    x_train = training_set[:, 0:4].T
    y_train_label = training_set[:, 4]
    onehot_y_train, Q = onehot_encoding(y_train_label)

    x_test = test_set[:, 0:4].T
    y_test_label = test_set[:, 4]
    onehot_y_test, _ = onehot_encoding(y_test_label)

    # 4. 하이퍼파라미터
    M = x_train.shape[0] - 1
    L = 10
    learning_rate = 0.0001
    epoch = 1000

    # 5. 초기 가중치 설정
    v, w = set_w_v(M, L, Q)

    # 6. 학습 (Accuracy + MSE 수집 포함)
    v, w, acc_list, mse_list = training_with_metrics(x_train, onehot_y_train, v, w, learning_rate, epoch)
    

    # 7. 그래프 출력
    plt.rcParams.update({'font.size': 20})
    plt.figure()
    plt.plot(range(1, epoch+1), acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Epoch')
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(range(1, epoch+1), mse_list)
    plt.xlabel('Epoch')
    plt.ylabel('MSE (Mean Squared Error)')
    plt.title('Training MSE per Epoch')
    plt.grid()
    plt.show()

# 8. 테스트 세트에 대한 예측 및 수동 Confusion Matrix 구현
    N_test = x_test.shape[1]
    # 실제 레이블 (0,1,2,…)
    y_true = y_test_label.astype(int)
    # 예측 레이블을 저장할 배열
    y_pred = np.zeros(N_test, dtype=int)

    # 8-1) forward_propagation으로 예측
    for i in range(N_test):
        x_n = x_test[:, i].reshape(-1, 1)  # (M+1,1)
        y_hat, _ = forward_propagation(x_n, v, w)  # (Q x 1)
        y_pred[i] = np.argmax(y_hat)               # 가장 높은 확률 인덱스

    # 8-2) 수동 Confusion Matrix 생성
    num_classes = max(y_true.max(), y_pred.max()) + 1
    cm_manual = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm_manual[t, p] += 1

        # 8-3) 결과 출력
    print("=== 수동 Confusion Matrix ===")
    print(cm_manual)

     # 8-5) Confusion Matrix 시각화 (row0·col6 제거 후)
    cm_plot = cm_manual[1:, :6]  # rows 1~6, cols 0~5

    plt.figure(figsize=(6,6))
    plt.imshow(cm_plot, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (hide row 0 & col 6)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # 컬러바
    cbar = plt.colorbar()
    cbar.set_label('Count', rotation=270, labelpad=15)

    # 눈금: 남은 행 인덱스 1~6 → y-label 1~6
    yticks = np.arange(cm_plot.shape[0])      # [0,1,2,3,4,5]
    plt.yticks(yticks, yticks+1)              # 1~6

    # 눈금: 남은 열 인덱스 0~5 → x-label 1~6 로 바꾸기
    xticks = np.arange(cm_plot.shape[1])      # [0,1,2,3,4,5]
    plt.xticks(xticks, xticks+1)              # 1~6

    # 셀마다 숫자 쓰기
    thresh = cm_plot.max() / 2
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            plt.text(j, i,
                     cm_plot[i, j],
                     ha="center", va="center",
                     color="white" if cm_plot[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()





    
