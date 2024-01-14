import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.transforms as transforms

from yaml import load, FullLoader
import time

from Loader import default_loader, is_image_file
from model.transMatrix import TransModule
from model.lib.matrix import MulLayer
from model.lib.libs import encoder3, encoder4, encoder5, decoder3, decoder4

def get_img(dataPath):
    fine_size = 256
    transform = transforms.Compose([
                transforms.Resize(fine_size),
                transforms.ToTensor()])
    Img = default_loader(dataPath)
    ImgA = transform(Img)
    return torch.unsqueeze(ImgA, 0)

#预测
def test(model_path, content_data_path, style_data_path, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    state_dict = torch.load(model_path)

    #加载设置和超参数
    print('-' * 20, 'loading config', '-' * 20)
    config = state_dict['config']
    print('finish loading config')
    config['device'] = device

    #加载数据
    print('-' * 20, 'loading data', '-' * 20)
    print('content_data_path:', content_data_path)
    print('style_data_path:', style_data_path)
    content_data = get_img(content_data_path)
    style_data = get_img(style_data_path)
    print('finish loading data')

    #加载模型
    print('-' * 20, 'loading model', '-' * 20)
    if config['layer'] == 'relu3_1':
        trans_module = TransModule('relu3_1')
        enc = encoder3()
        dec = decoder3()
        enc.load_state_dict(torch.load(config['pretrained_model_root'] + 'vgg_r31.pth'))
        dec.load_state_dict(torch.load(config['pretrained_model_root'] + 'dec_r31.pth'))
    elif config['layer'] == 'relu4_1':
        trans_module = TransModule('relu4_1')
        enc = encoder4()
        dec = decoder4()
        enc.load_state_dict(torch.load(config['pretrained_model_root'] + 'vgg_r41.pth'))
        dec.load_state_dict(torch.load(config['pretrained_model_root'] + 'dec_r41.pth'))
    trans_module.load_state_dict(state_dict['parameters'])
    print('finish loading model')

    contentV = torch.Tensor(1, 3, config['fine_size'], config['fine_size'])
    styleV = torch.Tensor(1, 3, config['fine_size'], config['fine_size'])

    enc.to(device)
    dec.to(device)
    trans_module.to(device)

    print('-' * 20, 'starting testing', '-' * 20)
    #预测模式
    trans_module.eval() 

    contentV.resize_(content_data.size()).copy_(content_data)
    contentV = contentV.to(device)
    styleV.resize_(style_data.size()).copy_(style_data)
    styleV = styleV.to(device)

    #forward
    with torch.no_grad():
        if config['layer'] == 'relu3_1':
            f_content, f_style = enc(contentV), enc(styleV)
        elif config['layer'] == 'relu4_1':
            f_content, f_style = enc(contentV)['relu4_1'], enc(styleV)['relu4_1']
        f_d, T = trans_module(f_content, f_style)
        predV = dec(f_d)

    predV = predV.clamp(0, 1)
    vutils.save_image(predV, save_path)
    print('Output image saved at {}'.format(save_path))