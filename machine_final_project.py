import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def k_mean(x, k, tol=1e-4, max_iter=300):
    """
    K-means Clustering을 수행하여 최종 중심(center_value)과 클러스터(assignments)를 반환합니다.
    입력:
      x: (N, M) 데이터 배열
      k: 클러스터 개수
      tol: 중심 이동 허용 오차
      max_iter: 최대 반복 횟수
    반환:
      centers: (k, M) 최종 중심
      assignments: (N,) 각 샘플의 클러스터 인덱스
    """
    N, M = x.shape
    # 1) 초기 중심 랜덤 선택
    indices = np.random.choice(N, k, replace=False)
    centers = x[indices].copy()

    for _ in range(max_iter):
        # 2) 할당: 각 샘플을 가장 가까운 중심으로
        dists = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)  # (N, k)
        assignments = np.argmin(dists, axis=1)  # (N,)

        # 3) 중심 업데이트
        new_centers = np.zeros_like(centers)
        for j in range(k):
            members = x[assignments == j]
            if len(members) > 0:
                new_centers[j] = members.mean(axis=0)
            else:
                # 빈 클러스터: 랜덤 재초기화
                new_centers[j] = x[np.random.randint(N)]

        # 4) 수렴 확인
        shift = np.linalg.norm(centers - new_centers, axis=1)
        centers = new_centers
        if np.max(shift) < tol:
            break

    # 최종 할당
    dists = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=2)
    assignments = np.argmin(dists, axis=1)
    return centers, assignments


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

# 그레이스케일 변환: RGB 채널 평균
def to_gray(img):
    return img.mean(axis=2).astype(np.float32)

# 1) 전체 그레이스케일 평균
def feature_1(input_data):
    gray = to_gray(input_data)
    return float(gray.mean())

# 2) 전체 그레이스케일 분산
def feature_2(input_data):
    gray = to_gray(input_data)
    return float(gray.var())

# 3) 가로축 투영 PDF → 기댓값
def feature_3(input_data):
    gray = to_gray(input_data)
    proj = gray.sum(axis=1)
    total = proj.sum()
    if total == 0: return 0.0
    pdf = proj / total
    idx = np.arange(proj.size, dtype=np.float32)
    return float((idx * pdf).sum())

# 4) 세로축 투영 PDF → 분산
def feature_4(input_data):
    gray = to_gray(input_data)
    proj = gray.sum(axis=0)
    total = proj.sum()
    if total == 0: return 0.0
    pdf = proj / total
    idx = np.arange(proj.size, dtype=np.float32)
    mean = (idx * pdf).sum()
    return float(((idx - mean)**2 * pdf).sum())

# 5) 주대각선 PDF → 기댓값
def feature_5(input_data):
    gray = to_gray(input_data)
    diag = np.diag(gray)
    total = diag.sum()
    if total == 0: return 0.0
    pdf = diag / total
    idx = np.arange(diag.size, dtype=np.float32)
    return float((idx * pdf).sum())

# 6) 주대각선 0 픽셀 비율
def feature_6(input_data):
    gray = to_gray(input_data)
    diag = np.diag(gray)
    return float(np.count_nonzero(diag == 0) / diag.size)

# 7) 히스토그램 PDF → 기댓값
def feature_7(input_data):
    gray = to_gray(input_data).flatten()
    hist, _ = np.histogram(gray, bins=28, range=(0,255))
    total = hist.sum()
    if total == 0: return 0.0
    pdf = hist / total
    idx = np.arange(hist.size, dtype=np.float32)
    return float((idx * pdf).sum())

# 8) X축 경계 강도 합
def feature_8(input_data):
    gray = to_gray(input_data)
    grad_x = np.abs(gray[:, 1:] - gray[:, :-1])
    return float(grad_x.sum())

# 9) Hu 모멘트 첫번째 값
def feature_9(input_data):
    gray = to_gray(input_data)
    H, W = gray.shape
    thresh = np.median(gray)
    bw = (gray > thresh).astype(np.float32)
    ys, xs = np.indices((H, W))
    m00 = bw.sum() + 1e-6
    x_mean = (xs * bw).sum() / m00
    y_mean = (ys * bw).sum() / m00
    mu20 = (((xs - x_mean)**2) * bw).sum()
    mu02 = (((ys - y_mean)**2) * bw).sum()
    nu20 = mu20 / (m00**2)
    nu02 = mu02 / (m00**2)
    return float(nu20 + nu02)

# 10) 객체 픽셀 비율
def feature_10(input_data):
    gray = to_gray(input_data)
    thresh = np.median(gray)
    bw = (gray > thresh)
    return float(bw.sum() / bw.size)


def select_features(directory):
    file_list = os.listdir(directory)
    features_list = []
    labels = []

    for name in file_list:
        path = os.path.join(directory, name)
        # 라벨: 파일명 앞 숫자
        labels.append(int(name.split('_', 1)[0]))

        # 이미지 읽기 및 RGB 변환
        img_BGR = cv2.imread(path)
        if img_BGR is None:
            print(f"Warning: failed to read {path}")
            continue
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

        # 10가지 특징 추출
        feats = [
            feature_1(img_RGB), feature_2(img_RGB), feature_3(img_RGB),
            feature_4(img_RGB), feature_5(img_RGB), feature_6(img_RGB),
            feature_7(img_RGB), feature_8(img_RGB), feature_9(img_RGB),
            feature_10(img_RGB)
        ]
        features_list.append(feats)

    # NumPy 배열로 변환
    features = np.array(features_list, dtype=np.float32)
    return features, labels


if __name__ == "__main__":
    # 1) 데이터 로드 및 특징 추출
    data_dir = r"C:\Users\sangh\train"
    features, labels = select_features(data_dir)  # (N, 10), list of labels

    # 2) 학습/테스트 셋 분할 (7:3) - features와 labels를 함께 섞고 분할
    N = features.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)
    features_shuf = features[indices]
    labels_shuf = np.array(labels)[indices]

    train_size = int(N * 0.7)
    X_train = features_shuf[:train_size]
    y_train = labels_shuf[:train_size]
    X_test  = features_shuf[train_size:]
    y_test  = labels_shuf[train_size:]

        # 3) 피처 정규화 (Raw level)
    # 학습 데이터의 평균·표준편차 계산
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0  # 분산 0 방지
    # 정규화
    X_train = (X_train - mu) / sigma
    X_test  = (X_test  - mu) / sigma

    # 4) 클러스터링 → 은닉층 가중치 초기화 → 은닉층 가중치 초기화
    M = X_train.shape[1]
    L = 70  # 예: M=10일 때 L=70
    centers, assignments = k_mean(X_train, L)

    # 4) 신경망 파라미터 초기화
    Q = 10
    v = np.zeros((L, M+1), dtype=np.float32)
    w = np.random.randn(Q, L+1).astype(np.float32)
    v[:, :M] = centers
    v[:,  M] = np.random.randn(L) * 0.01

    # 5) One-hot encoding
    y_train_onehot, _ = onehot_encoding(y_train)
    y_test_onehot,  _ = onehot_encoding(y_test)

    # 6) 학습 하이퍼파라미터
    epochs = 100
    lr = 0.01
    batch_size = 32

    train_loss = []
    train_acc = []    # 추가: Epoch별 Accuracy 저장
    train_mse = []    # 추가: Epoch별 MSE 저장
    test_acc = []   # 추가
    test_mse = []   # 추가
    for epoch in range(epochs):
        perm = np.random.permutation(train_size)
        X_shuf_epoch = X_train[perm]
        Y_shuf_epoch = y_train_onehot[perm]
        loss_accum = 0.0

        for i in range(0, train_size, batch_size):
            xb = X_shuf_epoch[i:i+batch_size]
            yb = Y_shuf_epoch[i:i+batch_size]

            # Forward
            xb_bias = np.hstack([xb, np.ones((xb.shape[0], 1))])
            alpha = xb_bias @ v.T
            b = sigmoid(alpha)
            b_bias = np.hstack([b, np.ones((b.shape[0], 1))])
            beta = b_bias @ w.T
            y_hat = sigmoid(beta)

            # Backward
            e = y_hat - yb
            grad_y = e * (y_hat * (1 - y_hat))
            delta_w = grad_y.T @ b_bias
            grad_b = (grad_y @ w[:, :-1]) * (b * (1 - b))
            delta_v = grad_b.T @ xb_bias

            # Update
            w -= lr * delta_w
            v -= lr * delta_v

            loss_accum += np.mean(e**2)

        train_loss.append(loss_accum)
        
        # Epoch 종료 후 전체 학습셋에 대해 Accuracy와 MSE 측정
        xb_all = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
        alpha_all = xb_all @ v.T
        b_all = sigmoid(alpha_all)
        b_all_bias = np.hstack([b_all, np.ones((b_all.shape[0], 1))])
        beta_all = b_all_bias @ w.T
        y_all_hat = sigmoid(beta_all)
        
        _, acc = Accuracy_max(y_train, y_all_hat)
        mse = compute_MSE(y_train_onehot, y_all_hat)
        
        train_acc.append(acc)
        train_mse.append(mse)
        
        # 테스트셋 평가
        xb_test_epoch = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
        alpha_test = xb_test_epoch @ v.T
        b_test = sigmoid(alpha_test)
        b_test_bias = np.hstack([b_test, np.ones((b_test.shape[0], 1))])
        beta_test = b_test_bias @ w.T
        y_test_hat = sigmoid(beta_test)

        _, acc_test_epoch = Accuracy_max(y_test, y_test_hat)
        mse_test_epoch = compute_MSE(y_test_onehot, y_test_hat)

        test_acc.append(acc_test_epoch)
        test_mse.append(mse_test_epoch)
   


    # 7) 테스트 평가
    xb_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    alpha_t = xb_test @ v.T
    b_t = sigmoid(alpha_t)
    b_t_bias = np.hstack([b_t, np.ones((b_t.shape[0], 1))])
    beta_t = b_t_bias @ w.T
    y_pred = sigmoid(beta_t)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    

    # 8) 평가 지표
    conf_mat = np.zeros((Q, Q), dtype=int)
    for true, pred in zip(y_test, y_pred_labels):
        conf_mat[true, pred] += 1
    accuracy = np.mean(y_pred_labels == y_test)
    final_test_mse = np.mean((y_pred - y_test_onehot) ** 2)

    print("Confusion Matrix:\n", conf_mat)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test MSE: {final_test_mse:.4f}")

   # 9) Confusion Matrix 시각화 (정규화 & 컬러바 포함)
    # —————————————————————————————————————————————
    class_names = [
        'green apple(0)', 'apple(1)', 'banana(2)', 'blackberry(3)', 'cucumber(4)',
        'orange(5)', 'peach(6)', 'pear(7)', 'tomato(8)', 'watermelon(9)'
    ]

    cm_norm = conf_mat.astype(float) / conf_mat.sum(axis=1, keepdims=True)

    fig1, ax1 = plt.subplots(figsize=(8, 8))
    im = ax1.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    cbar = fig1.colorbar(im, ax=ax1)
    cbar.ax.set_ylabel('Proportion', rotation=270, labelpad=15)

    ax1.set_xlabel('Predicted label')
    ax1.set_ylabel('True label')
    ax1.set_xticks(np.arange(len(class_names)))
    ax1.set_yticks(np.arange(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.set_yticklabels(class_names)

    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            ax1.text(j, i, f"{val:.3f}",
                     ha='center', va='center',
                     color='white' if val > thresh else 'black',
                     fontsize=10)

    ax1.set_title(f'Confusion Matrix (Accuracy {accuracy*100:.2f}%)')
    plt.show(block=False)   # 첫 번째 창 (Confusion Matrix)

    
        # 3) Train Accuracy
    fig3, ax3 = plt.subplots()
    ax3.plot(train_acc, label="Train Accuracy", linewidth=2)
    peak_idx = np.argmax(train_acc)
    peak_val = train_acc[peak_idx]
    ax3.plot(peak_idx, peak_val, 'ro')  # 빨간 점
    ax3.text(peak_idx, peak_val, f"{peak_val:.4f} (Peak)", fontsize=14, color='red',
             ha='center', va='bottom')
    ax3.set_title("Training Accuracy", fontsize=18)
    ax3.set_xlabel("Epoch", fontsize=18)
    ax3.set_ylabel("Accuracy", fontsize=18)
    ax3.grid(True)
    ax3.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show(block=False)
    
    # 4) Train MSE
    fig4, ax4 = plt.subplots()
    ax4.plot(train_mse, label="Train MSE", linewidth=2)
    min_idx = np.argmin(train_mse)
    min_val = train_mse[min_idx]
    ax4.plot(min_idx, min_val, 'ro')
    ax4.text(min_idx, min_val, f"{min_val:.4f} (Min)", fontsize=14, color='red',
             ha='center', va='bottom')
    ax4.set_title("Training MSE", fontsize=18)
    ax4.set_xlabel("Epoch", fontsize=18)
    ax4.set_ylabel("MSE", fontsize=18)
    ax4.grid(True)
    ax4.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show(block=False)
    
    # 5) Test Accuracy
    fig5, ax5 = plt.subplots()
    ax5.plot(test_acc, label="Test Accuracy", linewidth=2, color='orange')
    peak_idx = np.argmax(test_acc)
    peak_val = test_acc[peak_idx]
    ax5.plot(peak_idx, peak_val, 'ro')
    ax5.text(peak_idx, peak_val, f"{peak_val:.4f} (Peak)", fontsize=14, color='red',
             ha='center', va='bottom')
    ax5.set_title("Test Accuracy", fontsize=18)
    ax5.set_xlabel("Epoch", fontsize=18)
    ax5.set_ylabel("Accuracy", fontsize=18)
    ax5.grid(True)
    ax5.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show(block=False)
    
    # 6) Test MSE
    fig6, ax6 = plt.subplots()
    ax6.plot(test_mse, label="Test MSE", linewidth=2, color='red')
    min_idx = np.argmin(test_mse)
    min_val = test_mse[min_idx]
    ax6.plot(min_idx, min_val, 'ro')
    ax6.text(min_idx, min_val, f"{min_val:.4f} (Min)", fontsize=14, color='red',
             ha='center', va='bottom')
    ax6.set_title("Test MSE", fontsize=18)
    ax6.set_xlabel("Epoch", fontsize=18)
    ax6.set_ylabel("MSE", fontsize=18)
    ax6.grid(True)
    ax6.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show(block=False)


    # 11) 모델 가중치 CSV로 저장
    
    # v: 은닉층 가중치 (L x M+1)
    # w: 출력층 가중치 (Q x L+1)
    
    df_v = pd.DataFrame(v)
    df_v.to_csv('v_weights.csv', index=False, header=False)
    print(f"Saved v weights to {os.path.abspath('v_weights.csv')} (shape {v.shape})")
    
    df_w = pd.DataFrame(w)
    df_w.to_csv('w_weights.csv', index=False, header=False)
    print(f"Saved w weights to {os.path.abspath('w_weights.csv')} (shape {w.shape})")
    



