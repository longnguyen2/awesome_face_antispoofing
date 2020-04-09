#codin:utf8
from config import opt
import os
import models
import face_alignment
from skimage import io
from torch.autograd import Variable
from torchnet import meter
from utils import Visualizer
from tqdm import tqdm
from torchvision import transforms
import torchvision
import torch
from torchsummary import summary
import json
import numpy as np
import cv2
class DataHandle():

    def __init__(self,scale=2.7,image_size=224,use_gpu=False,transform=None,data_source = None):
        self.transform = transform
        self.scale = scale
        self.image_size = image_size
        if use_gpu:
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        else:
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,device='cpu')

    def det_img(self,image_data):
        input = image_data[:, :, ::-1]
        preds = self.fa.get_landmarks(input)
        # if 0:
        #     for pred in preds:
        #         img = cv2.imread(imgdir)
        #         print('ldmk num:', pred.shape[0])
        #         for i in range(pred.shape[0]):
        #             x,y = pred[i]
        #             print(x,y)
        #             cv2.circle(img,(x,y),1,(0,0,255),-1)
        #         cv2.imshow('-',img)
        #         cv2.waitKey()
        return preds
    def crop_with_ldmk(self,image, landmark):
        ct_x, std_x = landmark[:,0].mean(), landmark[:,0].std()
        ct_y, std_y = landmark[:,1].mean(), landmark[:,1].std()

        std_x, std_y = self.scale * std_x, self.scale * std_y

        src = np.float32([(ct_x, ct_y), (ct_x + std_x, ct_y + std_y), (ct_x + std_x, ct_y)])
        dst = np.float32([((self.image_size -1 )/ 2.0, (self.image_size -1)/ 2.0),
                  ((self.image_size-1), (self.image_size -1 )),
                  ((self.image_size -1 ), (self.image_size - 1)/2.0)])
        retval = cv2.getAffineTransform(src, dst)
        result = cv2.warpAffine(image, retval, (self.image_size, self.image_size), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT)
        return result

    def get_data(self,image_data):#第二步装载数据，返回[img,label]
        img = image_data
        ldmk = np.asarray(self.det_img(image_data),dtype=np.float32)
        # if 0:
        #     for pred in ldmk:
        #         for i in range(pred.shape[0]):
        #             x,y = pred[i]
        #             cv2.circle(img,(x,y),1,(0,0,255),-1)
        ldmk = ldmk[np.argsort(np.std(ldmk[:,:,1],axis=1))[-1]]
        img = self.crop_with_ldmk(img, ldmk)
        # if 0:
        #     cv2.imshow('crop face',img)
        #     cv2.waitKey()

        return np.transpose(np.array(img, dtype = np.float32), (2, 0, 1))

    def __len__(self):
        return len(self.img_label)
    def crop_with_ldmk(self,image, landmark):
        ct_x, std_x = landmark[:,0].mean(), landmark[:,0].std()
        ct_y, std_y = landmark[:,1].mean(), landmark[:,1].std()

        std_x, std_y = self.scale * std_x, self.scale * std_y

        src = np.float32([(ct_x, ct_y), (ct_x + std_x, ct_y + std_y), (ct_x + std_x, ct_y)])
        dst = np.float32([((self.image_size -1 )/ 2.0, (self.image_size -1)/ 2.0),
                  ((self.image_size-1), (self.image_size -1 )),
                  ((self.image_size -1 ), (self.image_size - 1)/2.0)])
        retval = cv2.getAffineTransform(src, dst)
        result = cv2.warpAffine(image, retval, (self.image_size, self.image_size), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT)
        return result

def main():
    data_handle = DataHandle(
                        scale = opt.cropscale,
                        use_gpu = opt.use_gpu,
                        transform = None,
                        data_source='none')
    import glob
    pths = glob.glob('checkpoints/%s/*.pth'%(opt.model))
    pths.sort(key=os.path.getmtime,reverse=True)
    print(pths)
    # 模型
    opt.load_model_path=pths[0]
    model = getattr(models, opt.model)().eval()
    assert os.path.exists(opt.load_model_path)
    if opt.load_model_path:
       model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()
    model.train(False)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while True:
        _, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            data = data_handle.get_data(img)
            data = data[np.newaxis,:]
            data = torch.FloatTensor(data)
            with torch.no_grad():
                if opt.use_gpu:
                    data =  data.cuda()
                outputs = model(data)
                outputs = torch.softmax(outputs,dim=-1)
                preds = outputs.to('cpu').numpy()
                attack_prob = preds[:,opt.ATTACK]
                if attack_prob > 0.5: cv2.putText(img, 'Fake', (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)
                else: cv2.putText(img, 'Real', (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
    cap.release()

if __name__=='__main__':
    main()
