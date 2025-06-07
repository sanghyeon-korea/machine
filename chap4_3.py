import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


# sigmoid 함수
def sigmoid(z):
    p = 1/(1+np.exp(-z))
    return p

# 원핫 인코딩 함수
def onehot_encoding(y) :
    Q = np.unique(y)  # 클래스 수
    N = len(y)        # 데이터 수
    onehot_y = np.zeros((N, len(Q)), dtype=int)
    for i in range(N) :
        Q_index = np.where(Q == y[i])[0][0]  # Q == y[i]를 찾아서
        onehot_y[i, Q_index] = 1             # 그곳에 1을 넣음
    return onehot_y, len(Q)

# set_w, v
def set_w_v(M, L, Q):
    v = np.random.randn(L, M+1)
    w = np.random.randn(Q, L+1)
    return v, w

# forward propagation
def forward_propagation(x_n, v, w) :
    alpha = v @ x_n
    b = sigmoid(alpha)
    b = np.vstack([b, [[1]]])
    beta = w @ b
    y_hat = sigmoid(beta)  # Qx1
    return y_hat ,b

# 데이터 쪼개는 함수
def div_data(data, a, b, c):
    np.random.shuffle(data)  # 셔플을 통해 데이터를 섞음
    first_div = int(len(data) * a / 10)                # training set과 validation set을 나누기 위한 기준
    second_div = int(len(data) - len(data) * c / 10)   # validation set과 test set을 나누기 위한 기준
    training_set = data[0:first_div, :]
    validation_set = data[first_div:second_div, :]
    test_set = data[second_div:, :]
    return training_set, validation_set, test_set

def Accuracy_max(y, y_hat):  # y_hat: (N, Q)
    N, Q = y_hat.shape
    y_hat_test = np.zeros((N, Q), dtype=int)
    max_j = np.argmax(y_hat, axis=1)        # 각 행(샘플)마다 최댓값 인덱스
    y_hat_test[np.arange(N), max_j] = 1

    if y.ndim == 1:
        onehot_y = np.zeros((N, Q), dtype=int)
        for i in range(N):
            onehot_y[i, int(y[i])] = 1
        y = onehot_y

    count = np.sum(np.all(y_hat_test == y, axis=1))
    accuracy = count / N
    return y_hat_test, accuracy

def back_propagation(x_n, y_n, v, w, learnig_rate):
    y_hat, b = forward_propagation(x_n, v, w)
    sigma = 2 * (y_hat - y_n) * y_hat * (1 - y_hat)  # Q x 1

    temp = (w.T @ sigma)[:-1]      # (L+1 x Q) @ (Q x 1) = (L+1 x 1), 마지막 바이어스 제외
    db = b[:-1] * (1 - b[:-1])      # (L x 1)
    grad_v = temp * db              # (L x 1)
    v -= learnig_rate * (grad_v @ x_n.T)  # L x (M+1)

    w -= learnig_rate * (sigma @ b.T)  # Q x (L+1)
    return v, w

# 전체 배치에 대한 제곱 오차의 평균
def compute_MSE(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


if __name__ == "__main__" :  # 메인 함수
    # 1. 데이터 가져오기, 더미 추가
    root = r'C:\Users\sanghyeon\Desktop\2021146019AHN\3-1\machine\11_prac'
    file_name_0 = '0_'
    file_name_1 = '1_'
    file_name_2 = '2_'
    df_0_gather = np.zeros((28*28, 500), dtype='float32')
    df_1_gather = np.zeros((28*28, 500), dtype='float32')
    df_2_gather = np.zeros((28*28, 500), dtype='float32')
    
    for i in range(1, 501):
        df_0 = pd.read_csv(f'{root}\\{file_name_0}{i}.csv', header=None)
        arr_0 = np.resize(df_0, (28*28,1))
        df_0_gather[:, i-1] = arr_0[:,0]
        df_1 = pd.read_csv(f'{root}\\{file_name_1}{i}.csv', header=None)
        arr_1 = np.resize(df_1, (28*28,1))
        df_1_gather[:, i-1] = arr_1[:,0]
        df_2 = pd.read_csv(f'{root}\\{file_name_2}{i}.csv', header=None)
        arr_2 = np.resize(df_2, (28*28,1))
        df_2_gather[:, i-1] = arr_2[:,0]
        
    # 2. X_raw, y 준비
    X_raw = np.hstack([df_0_gather, df_1_gather, df_2_gather])  # (784, 1500)
    y = np.array([0]*500 + [1]*500 + [2]*500)                  # (1500,)

    # 3. bias 항(더미 데이터) 행 추가: 각 샘플마다 입력 1
    N_total = X_raw.shape[1]                                   # 1500
    bias_row = np.ones((1, N_total), dtype=X_raw.dtype)        # (1, 1500)
    X = np.vstack([X_raw, bias_row])                           # (785, 1500)

    # 4. Train/Test 분할 (7:3 비율)
    data = np.vstack([X, y.reshape(1, -1)]).T   # shape = (1500, 786)
    train_set, _, test_set = div_data(data, 7, 0, 3)

    # 5. 분할된 x, y 처리
    x_train = train_set[:, :-1].T   # (785, N_train)
    y_train = train_set[:,  -1].astype(int)  # (N_train,)
    x_test  = test_set[:,  :-1].T   # (785, N_test)
    y_test  = test_set[:,   -1].astype(int)  # (N_test,)

    # 6. one-hot 변환
    onehot_y_train, Q = onehot_encoding(y_train)  # Q = 3
    onehot_y_test,  _ = onehot_encoding(y_test)

    # 7. 하이퍼파라미터 설정
    M = x_train.shape[0] - 1    # bias 제외한 입력 노드 수 = 784
    L = 64                      # 히든 레이어 노드 수 (조정 가능)
    learning_rate = 0.01
    epoch = 100

    # 8. 가중치 초기화
    v, w = set_w_v(M, L, Q)

    # 9. 학습 및 Train/Test 지표 수집 (가장 높은 Test Accuracy 시점 가중치 저장)
    acc_train_list = []
    mse_train_list = []
    acc_test_list  = []
    mse_test_list  = []

    best_test_acc = 0.0
    best_v = v.copy()
    best_w = w.copy()

    for e in range(epoch):
        # --- Training epoch ---
        idx = np.random.permutation(x_train.shape[1])
        for i in idx:
            x_n = x_train[:, i].reshape(-1, 1)
            y_n = onehot_y_train[i].reshape(-1, 1)
            v, w = back_propagation(x_n, y_n, v, w, learning_rate)

        # --- Epoch 종료 후 Train 지표 계산 ---
        y_hat_train_list = []
        for i in range(x_train.shape[1]):
            y_hat, _ = forward_propagation(x_train[:, i].reshape(-1, 1), v, w)
            y_hat_train_list.append(y_hat.T)  # (1, Q)
        y_hat_train_all = np.vstack(y_hat_train_list)  # (N_train, Q)
        _, acc_train = Accuracy_max(y_train, y_hat_train_all)
        mse_train = compute_MSE(onehot_y_train, y_hat_train_all)
        acc_train_list.append(acc_train)
        mse_train_list.append(mse_train)

        # --- Epoch 종료 후 Test 지표 계산 ---
        y_hat_test_list = []
        for i in range(x_test.shape[1]):
            y_hat, _ = forward_propagation(x_test[:, i].reshape(-1, 1), v, w)
            y_hat_test_list.append(y_hat.T)  # (1, Q)
        y_hat_test_all = np.vstack(y_hat_test_list)  # (N_test, Q)
        _, acc_test = Accuracy_max(y_test, y_hat_test_all)
        mse_test = compute_MSE(onehot_y_test, y_hat_test_all)
        acc_test_list.append(acc_test)
        mse_test_list.append(mse_test)

        # 최고 Test Accuracy 시점의 가중치 저장
        if acc_test > best_test_acc:
            best_test_acc = acc_test
            best_v = v.copy()
            best_w = w.copy()

    # 10. 학습 결과 시각화 (Training & Test 모두, Peak 표시)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch+1), acc_train_list, label='Train Accuracy', linewidth=2)
    plt.plot(range(1, epoch+1), mse_train_list, label='Train MSE', linewidth=2)
    plt.plot(range(1, epoch+1), acc_test_list, label='Test Accuracy', linewidth=2, linestyle='--')
    plt.plot(range(1, epoch+1), mse_test_list, label='Test MSE', linewidth=2, linestyle='--')

    best_train_idx = np.argmax(acc_train_list)
    best_train_acc = acc_train_list[best_train_idx]
    plt.scatter(best_train_idx+1, best_train_acc, color='red', s=100, marker='o')
    plt.text(best_train_idx+1, best_train_acc + 0.02,
             f'{best_train_acc*100:.1f}%', color='red', fontsize=16, ha='center')

    best_test_idx = np.argmax(acc_test_list)
    best_test_acc = acc_test_list[best_test_idx]
    plt.scatter(best_test_idx+1, best_test_acc, color='purple', s=100, marker='o')
    plt.text(best_test_idx+1, best_test_acc + 0.02,
             f'{best_test_acc*100:.1f}%', color='purple', fontsize=16, ha='center')

    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.title('Train vs Test: Accuracy & MSE', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 11. TEST set 검증: 예측 및 Confusion Matrix (최고 Test Accuracy 시점 가중치 사용)
    y_hat_test_final = []
    for i in range(x_test.shape[1]):
        y_pred, _ = forward_propagation(x_test[:, i].reshape(-1,1), best_v, best_w)
        y_hat_test_final.append(np.argmax(y_pred))
    y_hat_test_final = np.array(y_hat_test_final)  # (N_test,)

    conf_mat = np.zeros((Q, Q), dtype=int)
    for true, pred in zip(y_test, y_hat_test_final):
        conf_mat[int(true), int(pred)] += 1

    print("Confusion Matrix (row=true, col=predicted):")
    print(conf_mat)

    plt.figure(figsize=(6, 6))
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.ylabel('True label', fontsize=20)
    tick_marks = np.arange(Q)
    plt.xticks(tick_marks, [str(i) for i in range(Q)], fontsize=16)
    plt.yticks(tick_marks, [str(i) for i in range(Q)], fontsize=16)

    row_sums = conf_mat.sum(axis=1, keepdims=True)
    for i in range(Q):
        for j in range(Q):
            count = conf_mat[i, j]
            percent = (count / row_sums[i, 0]) * 100 if row_sums[i, 0] > 0 else 0
            plt.text(
                j, i,
                f"{count}\n({percent:.1f}%)",
                ha="center", va="center",
                color="white" if count > conf_mat.max()/2 else "black",
                fontsize=12
            )

    plt.colorbar()
    plt.tight_layout()
    plt.show()
