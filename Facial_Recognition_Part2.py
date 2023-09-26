import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = 'faces/'                                                        #얼굴 데이터 주소
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]    #얼굴 데이터 주소에 데이터 있을 시 하위 데이터 주소 저장

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]                                   #상위 데이터 주소와 하위 데이터 주소 저장
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)                   #학습을 위해 데이터 흑백 처리
    Training_Data.append(np.asarray(images, dtype=np.uint8))                #학습을 위해 Training_Data에 데이터 넣기
    Labels.append(i)                                                        #라벨 리스트에 인덱스 값 저장

Labels = np.asarray(Labels, dtype=np.int32)                                 #학습을 위해 assrray

model = cv2.face.LBPHFaceRecognizer_create()                                #얼굴 구별하기위한 학습 모델 생성

model.train(np.asarray(Training_Data), np.asarray(Labels))                  #학습시작

print("Model Training Complete!!!!!")


