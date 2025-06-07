import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


# sigmoid 함수
def sigmoid(z):
    p = 1/(1+np.exp(-z))
    return p

# 원핫 인코딩 함수
def onehot_encoding(y):
    Q = np.unique(y)  # 클래스 수
    N = len(y)        # 데이터 수
    onehot_y = np.zeros((N, len(Q)), dtype=int)
    for i in range(N):
        Q_index = np.where(Q == y[i])[0][0]
        onehot_y[i, Q_index] = 1
    return onehot_y, len(Q)

# set_w, v
def set_w_v(M, L, Q):
    v = np.random.randn(L, M+1)
    w = np.random.randn(Q, L+1)
    return v, w

# forward propagation
def forward_propagation(x_n, v, w):
    alpha = v @ x_n
    b = sigmoid(alpha)
    b = np.vstack([b, [[1]]])  # bias 항 추가
    beta = w @ b
    y_hat = sigmoid(beta)  # Qx1
    return y_hat, b

# 데이터 쪼개는 함수
def div_data(data, a, b, c):
    np.random.shuffle(data)
    first_div = int(len(data) * a / 10)
    second_div = int(len(data) - len(data) * c / 10)
    training_set = data[0:first_div, :]
    validation_set = data[first_div:second_div, :]
    test_set = data[second_div:, :]
    return training_set, validation_set, test_set

def Accuracy_max(y, y_hat):
    N, Q = y_hat.shape
    y_hat_test = np.zeros((N, Q), dtype=int)
    max_j = np.argmax(y_hat, axis=1)
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
    sigma = 2 * (y_hat - y_n) * y_hat * (1 - y_hat)
    temp = (w.T @ sigma)[:-1]
    db = b[:-1] * (1 - b[:-1])
    grad_v = temp * db
    v -= learnig_rate * (grad_v @ x_n.T)
    w -= learnig_rate * (sigma @ b.T)
    return v, w

def compute_MSE(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


# ====== 특징 추출 Case (feature 1 & feature 10만 사용) ======

# feature_1 함수: 가로축 Projection ⇒ PDF ⇒ 기댓값
def feature_1(input_data):
    proj = input_data.sum(axis=1).astype(np.float32)  # 길이 28
    total = np.sum(proj)
    if total == 0:
        return 0.0
    pdf = proj / total
    indices = np.arange(len(proj), dtype=np.float32)
    expect = np.sum(indices * pdf)
    return expect

# feature_10 함수: Anti-Diagonal ⇒ 0의 개수
def feature_10(input_data):
    anti = np.array([input_data[i, 27-i] for i in range(28)])
    return float(np.sum(anti == 0))


if __name__ == "__main__":
    root = r'C:\Users\sanghyeon\Desktop\2021146019AHN\3-1\machine\11_prac'
    file_name_0 = '0_'
    file_name_1 = '1_'
    file_name_2 = '2_'

    # 1. 데이터 가져오기 + 특징 추출
    x_0_set = np.empty((0, 2), dtype='float32')  # 1500×2

    for label in [0, 1, 2]:
        prefix = {0: file_name_0, 1: file_name_1, 2: file_name_2}[label]
        for i in range(1, 501):
            temp_path = f'{root}\\{prefix}{i}.csv'
            temp_image = pd.read_csv(temp_path, header=None).to_numpy(dtype='float32').reshape(28, 28)

            f1 = feature_1(temp_image)
            f10 = feature_10(temp_image)

            feat_vec = np.array([[f1, f10]], dtype='float32')  # (1×2)
            x_0_set = np.vstack((x_0_set, feat_vec))

    y_feat = np.array([0]*500 + [1]*500 + [2]*500)

    # 2. bias 항 추가
    X_raw_feat = x_0_set.T           # (2, 1500)
    N_total_feat = X_raw_feat.shape[1]  # 1500
    bias_row_feat = np.ones((1, N_total_feat), dtype=X_raw_feat.dtype)  # (1, 1500)
    X_feat = np.vstack([X_raw_feat, bias_row_feat])  # (3, 1500)

    # 3. Train/Test 분할 (7:3)
    data_feat = np.vstack([X_feat, y_feat.reshape(1, -1)]).T  # (1500, 4)
    train_set_feat, _, test_set_feat = div_data(data_feat, 7, 0, 3)

    x_train_feat = train_set_feat[:, :-1].T    # (3, N_train)
    y_train_feat = train_set_feat[:,  -1].astype(int)  # (N_train,)
    x_test_feat  = test_set_feat[:,  :-1].T    # (3, N_test)
    y_test_feat  = test_set_feat[:,   -1].astype(int)  # (N_test,)

    # 4. one-hot 인코딩
    onehot_y_train_feat, Q_feat = onehot_encoding(y_train_feat)  # Q_feat = 3
    onehot_y_test_feat,  _ = onehot_encoding(y_test_feat)

    # 5. 하이퍼파라미터 설정
    M_feat = x_train_feat.shape[0] - 1   # 입력 노드 수 = 2
    L_feat = 32                         # 히든 레이어 노드 수
    learning_rate_feat = 0.01
    epoch_feat = 100

    # 6. 가중치 초기화
    v_feat, w_feat = set_w_v(M_feat, L_feat, Q_feat)

    # 7. 학습 및 Train/Test 지표 수집 (최고 Test Accuracy 시점 가중치 저장)
    acc_train_list_feat = []
    mse_train_list_feat = []
    acc_test_list_feat  = []
    mse_test_list_feat  = []

    best_test_acc_feat = 0.0
    best_v_feat = v_feat.copy()
    best_w_feat = w_feat.copy()

    for e in range(epoch_feat):
        idx = np.random.permutation(x_train_feat.shape[1])
        for i in idx:
            x_n = x_train_feat[:, i].reshape(-1, 1)
            y_n = onehot_y_train_feat[i].reshape(-1, 1)
            v_feat, w_feat = back_propagation(x_n, y_n, v_feat, w_feat, learning_rate_feat)

        # Train 지표 계산
        y_hat_train_list_feat = []
        for i in range(x_train_feat.shape[1]):
            y_hat, _ = forward_propagation(x_train_feat[:, i].reshape(-1, 1), v_feat, w_feat)
            y_hat_train_list_feat.append(y_hat.T)
        y_hat_train_all_feat = np.vstack(y_hat_train_list_feat)  # (N_train, Q)
        _, acc_train_feat = Accuracy_max(y_train_feat, y_hat_train_all_feat)
        mse_train_feat = compute_MSE(onehot_y_train_feat, y_hat_train_all_feat)
        acc_train_list_feat.append(acc_train_feat)
        mse_train_list_feat.append(mse_train_feat)

        # Test 지표 계산
        y_hat_test_list_feat = []
        for i in range(x_test_feat.shape[1]):
            y_hat, _ = forward_propagation(x_test_feat[:, i].reshape(-1, 1), v_feat, w_feat)
            y_hat_test_list_feat.append(y_hat.T)
        y_hat_test_all_feat = np.vstack(y_hat_test_list_feat)  # (N_test, Q)
        _, acc_test_feat = Accuracy_max(y_test_feat, y_hat_test_all_feat)
        mse_test_feat = compute_MSE(onehot_y_test_feat, y_hat_test_all_feat)
        acc_test_list_feat.append(acc_test_feat)
        mse_test_list_feat.append(mse_test_feat)

        # 최고 Test Accuracy 시점의 가중치 저장
        if acc_test_feat > best_test_acc_feat:
            best_test_acc_feat = acc_test_feat
            best_v_feat = v_feat.copy()
            best_w_feat = w_feat.copy()

    # 8. 학습 결과 시각화 (Training & Test 모두, Peak 표시)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch_feat+1), acc_train_list_feat, label='Train Accuracy (feature)', linewidth=2)
    plt.plot(range(1, epoch_feat+1), mse_train_list_feat, label='Train MSE (feature)', linewidth=2)
    plt.plot(range(1, epoch_feat+1), acc_test_list_feat, label='Test Accuracy (feature)', linewidth=2, linestyle='--')
    plt.plot(range(1, epoch_feat+1), mse_test_list_feat, label='Test MSE (feature)', linewidth=2, linestyle='--')

    best_train_idx_feat = np.argmax(acc_train_list_feat)
    best_train_acc_feat = acc_train_list_feat[best_train_idx_feat]
    plt.scatter(best_train_idx_feat+1, best_train_acc_feat, color='red', s=100, marker='o')
    plt.text(best_train_idx_feat+1, best_train_acc_feat + 0.02,
             f'{best_train_acc_feat*100:.1f}%', color='red', fontsize=16, ha='center')

    best_test_idx_feat = np.argmax(acc_test_list_feat)
    best_test_acc_feat = acc_test_list_feat[best_test_idx_feat]
    plt.scatter(best_test_idx_feat+1, best_test_acc_feat, color='purple', s=100, marker='o')
    plt.text(best_test_idx_feat+1, best_test_acc_feat + 0.02,
             f'{best_test_acc_feat*100:.1f}%', color='purple', fontsize=16, ha='center')

    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Value', fontsize=20)
    plt.title('Train vs Test (feature): Accuracy & MSE', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 9. TEST set 검증: 예측 및 Confusion Matrix (최고 Test Accuracy 시점 가중치 사용)
    y_hat_test_final_feat = []
    for i in range(x_test_feat.shape[1]):
        y_pred, _ = forward_propagation(x_test_feat[:, i].reshape(-1, 1), best_v_feat, best_w_feat)
        y_hat_test_final_feat.append(np.argmax(y_pred))
    y_hat_test_final_feat = np.array(y_hat_test_final_feat)

    conf_mat_feat = np.zeros((Q_feat, Q_feat), dtype=int)
    for true, pred in zip(y_test_feat, y_hat_test_final_feat):
        conf_mat_feat[int(true), int(pred)] += 1

    print("Confusion Matrix (feature, row=true, col=predicted):")
    print(conf_mat_feat)

    plt.figure(figsize=(6, 6))
    plt.imshow(conf_mat_feat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (feature)', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.ylabel('True label', fontsize=20)
    tick_marks = np.arange(Q_feat)
    plt.xticks(tick_marks, [str(i) for i in range(Q_feat)], fontsize=16)
    plt.yticks(tick_marks, [str(i) for i in range(Q_feat)], fontsize=16)

    row_sums_feat = conf_mat_feat.sum(axis=1, keepdims=True)
    for i in range(Q_feat):
        for j in range(Q_feat):
            count = conf_mat_feat[i, j]
            percent = (count / row_sums_feat[i, 0]) * 100 if row_sums_feat[i, 0] > 0 else 0
            plt.text(
                j, i,
                f"{count}\n({percent:.1f}%)",
                ha="center", va="center",
                color="white" if count > conf_mat_feat.max()/2 else "black",
                fontsize=12
            )

    plt.colorbar()
    plt.tight_layout()
    plt.show()
