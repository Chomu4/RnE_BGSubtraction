import time

import cv2
import sys

import numpy as np

import u2net_wrapper

#from u2net_test import bg_sub
import csv

def make_noise(frame):
    frame = cv2.resize(frame, (640,360), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    frame = cv2.resize(frame, (1024,576), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    #noise = np.random.normal(2, 1, frame.shape).astype(np.uint8)
    #frame = cv2.add(frame, noise)
    h, w, c = frame.shape
    noisy_pixels = int(h * w * 0.01)
    # pepper = np.random.choice([0, 1], (h, w), p=[0.02, 0.98])
    # pepper = np.stack((pepper, pepper, pepper), axis=-1)
    '''
    for _ in range(noisy_pixels):
        row, col = np.random.randint(0, h), np.random.randint(0, w)
        if np.random.rand() < 0.5:
            frame[row, col] = [0, 0, 0]  # Pepper (black)
            pass
        else:
            # frame[row, col] = [255, 255, 255]  # Salt (white)
            pass
            # frame[row][col] = [numpy.float64(float(val) + set_noise) for val in (str(frame[row][col])[1:-1]).split(' ') if val]
    '''

    frame = frame.astype(int)
    frame = gaussian_noise(10, frame)

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
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture('test/simulation_result_20241212_231557.mp4')
    # cap = cv2.VideoCapture('test/test.mp4')

    if not cap.isOpened():
        print('Video open failed!')
        sys.exit()

    # u2net_wrapper.init(u2net_wrapper.Models.U2NET_HUMAN_SEG)

    # 배경 차분 알고리즘 객체 생성
    bs = cv2.createBackgroundSubtractorMOG2(history=110, detectShadows=False, varThreshold=16)
    #bs = cv2.createBackgroundSubtractorKNN(history=3) # 배경영상이 업데이트 되는 형태가 다름
    bs.setDetectShadows(False) # 그림자 검출 안하면 0과 255로 구성된 마스크 출력
    frame_n = 0
    data_mog = []
    data_u2net = []
    # 비디오 매 프레임 처리
    while True:
        t = time.time_ns()
        # print("loop")
        ret, frame = cap.read()
        if not ret:
            break

        frame_n += 1
        if frame_n % 60 != 0:
            continue

        try:
            frame = cv2.resize(frame, (1024, 576))
            cv2.imshow('frame', frame)
        except cv2.error:
            break

        frame = make_noise(frame)
        #fgmask = bg_sub(frame) # sample = {'imidx':imidx, 'image':image, 'label':label}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 0또는 128또는 255로 구성된 fgmask 생성
        #fgmask_noise = bs.apply(gray, learningRate=0)
        removed_noise = cv2.fastNlMeansDenoisingColored(frame, None, 15, 7, 21)
        t = time.time_ns()
        fgmask_noiseless = bs.apply(removed_noise, learningRate=0)
        # fgmask_noiseless_u2net = u2net_wrapper.u2net_bg_sub(removed_noise)
        back = bs.getBackgroundImage()
        # 배경 영상 받아오기

        cv2.imshow('frame_with_noise', frame)
        cv2.imshow('back', back)
        #cv2.imshow('fgmask_noise', fgmask_noise)

        cv2.imshow('removed_noise', removed_noise)

        cv2.imshow('fgmask_noiseless', fgmask_noiseless)
        # cv2.imshow('fgmask_noiseless_u2net', fgmask_noiseless_u2net)

        #mask = bs.apply(frame, learningRate=0)
        #cv2.imshow('mask', mask)
        cnt_mog = np.sum(fgmask_noiseless == 0)
        # cnt_u2net = np.sum(fgmask_noiseless_u2net < 60)
        # for i in range(frame.shape[0]):
        #     for j in range(frame.shape[1]):
        #         pixel_mog = fgmask_noiseless[i][j]
        #         pixel_u2net = fgmask_noiseless_u2net[i][j]
        #         if np.all(pixel_mog == 0):   # 검은색의 비율
        #             cnt_mog += 1
        #         if np.all(pixel_u2net < 60):
        #             cnt_u2net += 1

        data_mog.append([(round(cnt_mog/(640*480)*100))])
        # data_u2net.append([(round(cnt_u2net / (640 * 480) * 100))])


        # if cv2.waitKey(20) == 27:
        #     break
        key = cv2.waitKey(0)
        if key % 256 == 27:
            break
        dt = time.time_ns() - t
        print(f'Loop took {dt/1_000_000.0} ms to run')

    cap.release()
    cv2.destroyAllWindows()

    f = open("MOG2_result.csv", "w")
    csv.writer(f).writerows(data_mog)
    f.close()
    # f = open("U-2-Net_result.csv", "w")
    # csv.writer(f).writerows(data_u2net)
    # f.close()

def test():
    u2net_wrapper.init(u2net_wrapper.Models.U2NET)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        u2net_result = u2net_wrapper.u2net_bg_sub(frame)
        cv2.imshow("asdf", u2net_result)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()
    # test()