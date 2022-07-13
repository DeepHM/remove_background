import os
import os.path as osp
from tqdm import tqdm
import argparse

import random
import numpy as np
import cv2
import io
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt

from alpha import alpha_matting_cutout
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.rembg.u2net import data_loader,u2net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device :',device)

def preprocess(image):
    label_3 = np.zeros(image.shape)
    label = np.zeros(label_3.shape[0:2])

    if 3 == len(label_3.shape):
        label = label_3[:, :, 0]
    elif 2 == len(label_3.shape):
        label = label_3

    if 3 == len(image.shape) and 2 == len(label.shape):
        label = label[:, :, np.newaxis]
    elif 2 == len(image.shape) and 2 == len(label.shape):
        image = image[:, :, np.newaxis]
        label = label[:, :, np.newaxis]

    transform = transforms.Compose(
        [data_loader.RescaleT(320), data_loader.ToTensorLab(flag=0)]
    )
    sample = transform({"imidx": np.array([0]), "image": image, "label": label})

    return sample


def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)

    return dn

def naive_cutout(img, mask):
#     empty = Image.new("RGBA", (img.size), 0)  # 원본 이미지 size로 empty 생성
    empty = Image.new("RGB", (img.size), (255, 255, 255))

    ## Image.composite : Image 합성 함수
    # 리사이즈 방법에는 크게 bilinear, bicubic, lanczos 이렇게 세 가지가 있습니다.
    # bilinear로 리사이즈 할 경우 용량이 더 작아지고 인코딩 속도도 빠르지만 흐릿한 느낌을 주는 반면,
    # lanczos 방식은 용량도 커지고 인코딩 속도도 느리지만 가장 선명한 화질을 보여줍니다.
    #  ->  Lanczos. 일반적으로 지원되는 알고리즘 중 가장 고품질의 이미지를 얻을 수 있다.
    #  - > Lanczos. 푸리에와 비슷한 아이디어로 처리되는듯한 sinc ??
    # bicubic은 용량, 속도, 선명함에서 중간 정도라고 보시면 될 듯 합니다.
    cutout = Image.composite(img, empty, mask.resize(img.size, Image.LANCZOS))  # Lanzos : 이미지 보간 BILINEAR 같은 method임
    return cutout


def remove_background(img_dir,model,output_dir,use_gpu,alpha_matting) :
    
    f = np.fromfile(img_dir)
    img = Image.open(io.BytesIO(f)).convert("RGB")
    sample = preprocess(np.array(img))

    with torch.no_grad():
        if use_gpu:
            inputs_test = torch.cuda.FloatTensor(
                sample["image"].unsqueeze(0).cuda().float()
            )
        else:
            inputs_test = torch.FloatTensor(sample["image"].unsqueeze(0).float())

        d1, d2, d3, d4, d5, d6, d7 = model(inputs_test)

        pred = d1[:, 0, :, :]
        predict = norm_pred(pred)

        predict = predict.squeeze()
        predict_np = predict.cpu().detach().numpy()
        mask = Image.fromarray(predict_np * 255).convert("RGB")
        mask = mask.convert('L')

        del d1, d2, d3, d4, d5, d6, d7, pred, predict, predict_np, inputs_test, sample
        
    if alpha_matting:
        try : 
            print(' USE : alpha_matting! ')
            cutout = alpha_matting_cutout(
                img,
                mask,
                foreground_threshold=240,
                background_threshold=10,
                erode_structure_size=15,   #    "-ae",
                base_size=1000
                )
        except Exception:
            cutout = naive_cutout(img, mask)
    else:
        cutout = naive_cutout(img, mask)    
    
    result = cutout
        
#     result = np.array(cutout)
#     result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
#     save_file_name = img_dir.split('/')[-1]
#     save_file_name = save_file_name.split('.')[0]+'.png'
#     cv2.imwrite(osp.join(output_dir,save_file_name),result)
#     print('Prediction Result Save :',osp.join(output_dir,img_dir.split('.')[0]+'.png'))
    
    return result

def main(args) :
    
    ### Settings
    use_gpu = args.use_gpu   # True False
    alpha_matting = args.am  # True False
    input_dir = args.id
    output_dir = args.od
    model_dir = args.md
    model_name = args.model_name   ##  'UNET' or 'UNETP'
    assert model_name in ['UNET','UNETP'], "[Check] Available Model !!"
    ###
    
    ### Images
    imgs = os.listdir(input_dir)
    imgs = [i for i in imgs if i.split('.')[-1].lower() in ['jpg','jpeg','png']]
    ###
    
    ### Model define
    print('model : ',model_name)
    if model_name =='UNETP' :
        model = u2net.U2NETP(3, 1)
        model_file = osp.join(model_dir,'rembg_UNETPmodel2021_11_25.pt')
    elif model_name =='UNET' :
        model = u2net.U2NET(3, 1)
        model_file = osp.join(model_dir,'rembg_UNETmodel2021_11_25.pt')

    model.to(device)
    ## Basically, the [GPU-Memory] allocated by the model -> 2000MB
    model.load_state_dict(torch.load(model_file)['model_state_dict'])
    model.eval()
    ###


    test_bar = tqdm(range(len(imgs))) 
    for i in test_bar :
        img_path = osp.join(input_dir,imgs[i])
        result = remove_background(img_dir=osp.join(img_path),model=model,output_dir=output_dir,use_gpu=use_gpu,alpha_matting=alpha_matting)
        
        save_file_name = img_path.split('/')[-1]
        save_file_name = save_file_name.split('.')[0]+'.png'
        result.save(osp.join(output_dir,save_file_name), 'png')
        print('Prediction Result Save :',osp.join(output_dir,save_file_name))

        
        
    print(' @@ Finish @@ ' )



if __name__=="__main__" :
    parser = argparse.ArgumentParser(description='remove background')
    parser.add_argument('--use_gpu', action='store_true', default=True,
                    help='use_gpu')
    parser.add_argument('--am', action='store_true', default=False,
                    help='USE : alpha_matting')    
    parser.add_argument('--id',type=str,default='TEST_IMGS',
                    help='input_dir')    
    parser.add_argument('--od',type=str, default='RESULTS',
                    help='output_dir')    
    parser.add_argument('--md',type=str, default='model_files',
                    help='model_dir')   
    parser.add_argument('--model_name',type=str, default='UNETP',
                    help='model_name')  
    args = parser.parse_args()
    
    main(args)
    














