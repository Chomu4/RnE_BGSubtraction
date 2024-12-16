import enum
import os

import cv2
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
#import modell
# normalize the predicted SOD probability map


class Models(enum.StrEnum):
    U2NET = "u2net"
    U2NETP = "u2netp"

net: U2NET | U2NETP | None = None
dev = None

def init(model: Models):
    global net, dev
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")

    torch.set_grad_enabled(False)

    model_dir = os.path.join(os.getcwd(), 'saved_models', model, model + '.pth')

    if model == Models.U2NET:
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif model == Models.U2NETP:
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)
    else:
        raise "bro"

    net.load_state_dict(torch.load(model_dir, map_location=torch.device(dev)))
    net = net.to(device=dev)
    net.eval()

'''
def save_output(frame,pred):
    global net

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    # img_name = image_name.split(os.sep)[-1]
    image = frame
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.Resampling.BILINEAR)

    pb_np = np.array(imo)
    return cv2.cvtColor(pb_np, cv2.COLOR_RGB2BGR)
    # aaa = img_name.split(".")
    # bbb = aaa[0:-1]
    # imidx = bbb[0]
    # for i in range(1,len(bbb)):
    #     imidx = imidx + "." + bbb[i]
    #
    # imo.save(d_dir+imidx+'.png')
'''

IMAGE_INPUT_SIZE = 320

def preprocess(img: np.array) -> torch.Tensor:
    # Image resize
    img = transform.resize(img, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE), mode='constant')

    # RGB normalization
    tmp_img = np.zeros((img.shape[0], img.shape[1], 3))
    img = img / np.max(img)
    if img.shape[2] == 1:
        tmp_img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
        tmp_img[:, :, 1] = (img[:, :, 0] - 0.485) / 0.229
        tmp_img[:, :, 2] = (img[:, :, 0] - 0.485) / 0.229
    else:
        tmp_img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
        tmp_img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        tmp_img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

    tmp_img = tmp_img.transpose((2, 0, 1))
    torch_img = torch.from_numpy(tmp_img)

    return torch_img.float().unsqueeze(0)

def normalize_prediction(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def u2net_bg_sub(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    torch_frame = preprocess(frame).to(device=dev)
    res: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = net(torch_frame)
    d1, d2, d3, d4, d5, d6, d7 = res

    predict = d1[:, 0, :, :]
    predict = normalize_prediction(predict)

    predict = predict.squeeze()
    predict_np = predict.to(device="cpu").detach().numpy()

    predict_rgb = Image.fromarray(predict_np * 255).convert('RGB')
    predict_resize = predict_rgb.resize((frame.shape[1], frame.shape[0]), resample=Image.Resampling.BILINEAR)

    return cv2.cvtColor(np.array(predict_resize), cv2.COLOR_RGB2BGR)

'''
def bg_sub(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])(frame)


    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(
                                        frame=frame,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)



    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        # print("inferencing:",img_name_list[i_test].split(os.sep)[-1])
        print("inferencing")

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)
        
        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test],pred,prediction_dir)
        
        del d1,d2,d3,d4,d5,d6,d7
        return save_output(frame, pred)
"""
if __name__ == "__main__":
    main()
"""
'''