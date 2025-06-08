import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
밝기 특성만 이용한 이유 : 
배경이 흰색 또는 통일됨

이미지가 가운데 정렬되고 크기도 유사함

과일/채소의 밝기, 모양, 위치가 분류에 충분히 강력한 단서'''

# 그레이스케일 변환: RGB 채널 평균 -> 밝기 정보만 사용하기 위해 사용함
#밝을 수록 값이 커짐 -> 흰색인 배경의 값이 가장 큼
#불필요한 색정보 제거
def to_gray(img):
    return img.mean(axis=2).astype(np.float32)

# 1) 전체 그레이스케일 평균 -> 이미지 전체의 픽셀 밝기의 평균을 알기 위해 사용.
#얼마나 밝은 이미지인지 나타냄
#과일과 채소의 종류에 따라 밝기의 차이가 분명히 있기 때문에 강력한 특징이라 생각하여 사용
#예를 들면 수박은 어두운데 바나나는 밝음
def feature_1(input_data):
    gray = to_gray(input_data)
    return float(gray.mean())

# 2) 전체 그레이스케일 분산
'''밝기의 분산 -> 대비 정도를 나타내기 위해 사용. 색의 밝기 차이가 거의 없을 떄는 분산이 낮지만
밝기 차이가 큰 과일은 분산이 높다. 밝기 차이가 높은 과일은 질감이 좀 복잡한 과일 예를 들어 
블루베리가 있다.'''
def feature_2(input_data):
    gray = to_gray(input_data)
    return float(gray.var())

# 3) 가로축 투영 PDF → 기댓값
'''행마다의 밝기의 합을 사용. 밝기의 질량이 이미지 위, 아래 어디에 집중됐는지에 대한 특징을 추출하기 위한 함수 '''
def feature_3(input_data):
    gray = to_gray(input_data)
    proj = gray.sum(axis=1)
    total = proj.sum()
    if total == 0: return 0.0
    pdf = proj / total
    idx = np.arange(proj.size, dtype=np.float32)
    return float((idx * pdf).sum())

# 4) 세로축 투영 PDF → 분산
'''세로 방향의 밝기 분포의 분산, 좌우 방향으로 얼마나 넓게 퍼져 있는지에 대한 특징을 추출하기 위한 함수'''
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
'''왼쪽 맨 위부터 오른쪽 맨 아래 까지 대각선으로 이어진 곳의 밝기 분포의 평균 위치, 대각선 방향의 
밝기 질량 분포에 대한 특징을 추출하기 위한 삼수
가로 폭이 좁은 과일과 넓은 과일을 분류 가능'''
def feature_5(input_data):
    gray = to_gray(input_data)
    diag = np.diag(gray)
    total = diag.sum()
    if total == 0: return 0.0
    pdf = diag / total
    idx = np.arange(diag.size, dtype=np.float32)
    return float((idx * pdf).sum())

# 6) 주대각선 0 픽셀 비율
'''대각선에 있는 픽셀 중 밝기 = 0 의 비율을 특징 추출, 대각선 방향의 비어있는 정도를 확인하고 특징 추출
배경이 비어있고 중심에 물체가 있는 구조에서 좋은 분류가 가능하다고 생각하기에 사용
비어있는 경우 값이 크고 꽉 차 있는 경우 값이 작다.'''
def feature_6(input_data):
    gray = to_gray(input_data)
    diag = np.diag(gray)
    return float(np.count_nonzero(diag == 0) / diag.size)

# 7) 히스토그램 PDF → 기댓값
'''히스토그램 : 밝기 값이 얼마나 나왔는지에 대한 그래프
흑백 -> 0, 흰색 -> 255
즉 밝은 색이 많은 이미지일 수록 히스토그램이 오른쪽으로 쏠리고 어두울수록 왼쪽으로 쏠린다
'''
def feature_7(input_data):
    gray = to_gray(input_data).flatten()
    hist, _ = np.histogram(gray, bins=28, range=(0,255)) #0~255를 28개의 단계로 구간을 나눔
    total = hist.sum()
    if total == 0: return 0.0
    pdf = hist / total
    idx = np.arange(hist.size, dtype=np.float32)
    return float((idx * pdf).sum()) #결국 밝기값 x 확률이므로 히스토그램 중심이 어디있는지를 확인하여 특징 추출

# 8) X축 경계 강도 합
'''이미지에서 밝기가 갑자기 바뀌는 곳을 확인하기 위해 사용
예를들어 흰 배경에서 갑자기 다른 색으로 넘어가면 그곳에서의 밝기가 급격하게 바뀐다.
즉 물제의 외곽선을 파악할 수 있다. 사용되는 사진의 배경은 전부 흰색이기에 경계가 뚜렸하여
가장 효율적인 특징 추출이라고 생각함.'''
def feature_8(input_data):
    gray = to_gray(input_data)
    grad_x = np.abs(gray[:, 1:] - gray[:, :-1])
    return float(grad_x.sum()) #경계가 많은 이미지일 수록 값이 커지도록 설정

# 9) Hu 모멘트 첫번째 값
'''객체의 모양을 크기나 위치에 관계없이 특징 추출하기 위해 사용
형태 기반 분류'''
def feature_9(input_data):
    gray = to_gray(input_data)
    H, W = gray.shape
    thresh = np.median(gray)
    bw = (gray < thresh).astype(np.float32) #객체만 남김
    ys, xs = np.indices((H, W))
    m00 = bw.sum() + 1e-6
    x_mean = (xs * bw).sum() / m00 #객체의 중심 x좌표
    y_mean = (ys * bw).sum() / m00 #객체의 중심 y 좌표
    mu20 = (((xs - x_mean)**2) * bw).sum() #x축으로 얼마나 퍼졌느닞
    mu02 = (((ys - y_mean)**2) * bw).sum() #y축으로 얼마나 퍼졌는지
    nu20 = mu20 / (m00**2) #위치와 면적 영향을 제거함
    nu02 = mu02 / (m00**2) 
    return float(nu20 + nu02) 

# 10) 객체 픽셀 비율
'''이미지 전체에서 밝은 부분의 비율을 측정하기 위함
물체의 상대적 면적을 측정한다. 화면을 크게 차지한 물체는 값이 크고 
작은 경우 값이 작게 특징이 추출된다'''
def feature_10(input_data):
    gray = to_gray(input_data)
    thresh = np.median(gray)
    bw = (gray < thresh)
    return float(bw.sum() / bw.size)

#특성 선택, 데이터 전처리
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