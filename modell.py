import cv2
import sys

import numpy as np

import u2net_test
from u2net_test import bg_sub


#from u2net_test import bg_sub
import csv

def make_noise(frame):
    # frame = cv2.resize(frame, (160,120), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    # frame = cv2.resize(frame, (640,480), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    #noise = np.random.normal(2, 1, frame.shape).astype(np.uint8)
    #frame = cv2.add(frame, noise)
    h, w, c = frame.shape
    noisy_pixels = int(h * w * 0.01)
    # pepper = np.random.choice([0, 1], (h, w), p=[0.02, 0.98])
    # pepper = np.stack((pepper, pepper, pepper), axis=-1)

    for _ in range(noisy_pixels):
        row, col = np.random.randint(0, h), np.random.randint(0, w)
        if np.random.rand() < 0.5:
            frame[row, col] = [0, 0, 0]  # Pepper (black)
            pass
        else:
            # frame[row, col] = [255, 255, 255]  # Salt (white)
            pass
            # frame[row][col] = [numpy.float64(float(val) + set_noise) for val in (str(frame[row][col])[1:-1]).split(' ') if val]


    frame = frame.astype(int)
    frame = gaussian_noise(20, frame)

    # frame *= pepper

    return frame.astype(np.uint8)

def gaussian_noise(scale, frame):
    make_noise = np.random.normal(0, scale, frame.shape)  # 랜덤함수를 이용하여 노이즈 적용
    set_noise = (1 * make_noise).astype(int)
    frame += set_noise
    frame = np.clip(frame, 0, 255)
    return frame


def main():
    # 비디오 파일 열기
    # cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap=cv2.VideoCapture('test.mp4')

    if not cap.isOpened():
        print('Video open failed!')
        sys.exit()

    u2net_test.init()

    # 배경 차분 알고리즘 객체 생성
    bs = cv2.createBackgroundSubtractorMOG2(history=0, detectShadows=False, varThreshold=100)
    #bs = cv2.createBackgroundSubtractorKNN(history=3) # 배경영상이 업데이트 되는 형태가 다름
    bs.setDetectShadows(False) # 그림자 검출 안하면 0과 255로 구성된 마스크 출력
    frame_n = 0
    data = [['cv']]
    # 비디오 매 프레임 처리
    while True:
        frame_n += 1
        if frame_n % 60 != 0:
            continue
        ret, frame = cap.read()
        try:
            cv2.imshow('frame', frame)
        except cv2.error:
            break
        if not ret:
            break
        frame = make_noise(frame)
        #fgmask = bg_sub(frame) # sample = {'imidx':imidx, 'image':image, 'label':label}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 0또는 128또는 255로 구성된 fgmask 생성
        #fgmask_noise = bs.apply(gray, learningRate=0)
        removed_noise = cv2.fastNlMeansDenoising(frame, None, 30, 7, 21)
        fgmask_noiseless = bs.apply(removed_noise, learningRate=0)
        fgmask_noiseless_u2net = u2net_test.bg_sub(removed_noise)
        back = bs.getBackgroundImage()
        # 배경 영상 받아오기

        cv2.imshow('frame_with_noise', frame)
        cv2.imshow('back', back)
        #cv2.imshow('fgmask_noise', fgmask_noise)

        cv2.imshow('removed_noise', removed_noise)

        cv2.imshow('fgmask_noiseless', fgmask_noiseless)
        cv2.imshow('fgmask_noiseless_u2net', fgmask_noiseless_u2net)

        #mask = bs.apply(frame, learningRate=0)
        #cv2.imshow('mask', mask)
        if frame_n >= 10:
            frame_n = 0
            cnt = 0
            for i in range(frame.shape[0]):
                for j in range(frame.shape[1]):
                    pixel = fgmask_noiseless[i][j]
                    if np.all(pixel == 0):
                        cnt += 1
            data.append([(int(cnt/(640*480)*100))])


        if cv2.waitKey(20) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    f = open("result.csv", "w")
    csv.writer(f).writerows(data)
    f.close()

if __name__ == "__main__":
    main()