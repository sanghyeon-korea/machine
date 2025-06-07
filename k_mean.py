import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def k_mean_init(x, k) :
    random_number_store = []
    center_value = np.zeros((k ,x.shape[1] ))
    
    for i in range(k) :
        random_number = np.random.randint(0, len(x))
        random_number_store.append(random_number)
        center_value[i, :] = x[random_number, :]
        
    random_number_store = np.array(random_number_store)    
    return center_value, random_number_store 
    

def k_mean(k) :
    """
    K-means Clustering을 수행하여 최종 중심(center_value)과 클러스터(assignments)를 반환합니다.
    전역 변수 x를 사용합니다.
    """
    # 1) 초기 중심 가져오기
    center_value, _ = k_mean_init(x, k)

    # 반복 수행
    while True:
        # 2) 할당: 각 포인트를 가장 가까운 중심에 매핑
        assignments = np.zeros(len(x), dtype=int)
        for idx_point in range(len(x)):
            # L2 Norm 수동 계산
            dists = []
            for idx_cent in range(k):
                diff = x[idx_point] - center_value[idx_cent]
                # ||v||_2 = sqrt(sum(v_i^2))
                dist = np.sqrt((diff * diff).sum())
                dists.append(dist)
            assignments[idx_point] = int(np.argmin(dists))

        # 3) 중심 업데이트
        new_centers = np.zeros_like(center_value)
        for idx_cent in range(k):
            # 클러스터에 속한 포인트 모으기
            members = [x[i] for i in range(len(x)) if assignments[i] == idx_cent]
            if len(members) > 0:
                members_arr = np.array(members)
                # 차원별 평균 계산
                new_centers[idx_cent] = members_arr.mean(axis=0)
            else:
                # 빈 클러스터는 랜덤 재초기화
                rnd = np.random.randint(0, len(x))
                new_centers[idx_cent] = x[rnd]

        # 4) 수렴 확인 (중심 이동 거리)
        shifts = np.zeros(k)
        for i in range(k):
            diff = center_value[i] - new_centers[i]
            shifts[i] = np.sqrt((diff * diff).sum())

        # 모두 이동 거리가 1e-4 미만이면 종료
        if np.max(shifts) < 1e-4:
            center_value = new_centers
            break

        center_value = new_centers

    return center_value, assignments


if __name__ == "__main__":
    # 1. 데이터 가져오기 (첫 행 헤더)
    df = pd.read_csv('C:\\Users\\sanghyeon\\Desktop\\2021146019AHN\\3-1\\machine\\Clustering_data.csv')
    x = df.to_numpy()

    # 2. 초기 중심 선택 후 시각화
    init_centers, rnd_idx = k_mean_init(x, 3)
    plt.figure(figsize=(6,6))
    plt.scatter(x[:,0], x[:,1], s=20)
    plt.scatter(init_centers[:,0], init_centers[:,1], c='red', marker='X', s=100)
    plt.title('Initial Data and Centroids')
    plt.grid(True)
    plt.show()

    # 3. K-means 수행 (K=3)
    final_centers, assignments = k_mean(3)

    # 4. 최종 클러스터링 결과 시각화
    plt.figure(figsize=(6,6))
    for ci in range(3):
        pts = x[assignments == ci]
        plt.scatter(pts[:,0], pts[:,1], s=20, label=f'Cluster {ci}')

    plt.scatter(final_centers[:,0], final_centers[:,1], c='magenta', marker='X', s=150, label='Final Centroids')
    plt.title('Final K-means Clustering')
    plt.legend()
    plt.grid(True)
    plt.show()

