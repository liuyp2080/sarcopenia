
import pydicom
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from monai.data import ITKReader
from monai.transforms import (Compose,LoadImage,Resize,ScaleIntensityRange,ToTensor)
import streamlit as st
from io import BytesIO
import statsmodels.api as sm
import pingouin as pg
from sar_model import build_resunetplusplus
from scipy import stats
 
#ST用户界面 
st.set_page_config(page_title='Body Composition Measurement Tool', page_icon='🖼️',layout="centered", initial_sidebar_state="auto", menu_items=None)
st.header(':frame_with_picture: show them all')
st.write("**Introduction:**")
st.write("After all the calculation, you may want to observe the imgage with it's mask and label with you eyes to check whether they are match or not.")
st.write("First step: upload all the labels; Second step: upload all the CT images, and then you wait.**Note:**make sure the image and the label have the same name!")
"----"
st.header('Input')
col1,col2=st.columns([0.5,0.5])
with col1:
    #上传label文件
    uploaded_label=st.file_uploader(label='Please select **label** file(multiple allowed)',accept_multiple_files=True)
    #BUTTON 
    clear_button=st.button('Clear the uploaded files')
with col2:
    uploaded_images=st.file_uploader(label='Please select **CT** images(multiple allowed)',accept_multiple_files=True)
"----"
preprocess_dcm= Compose([
    LoadImage(dtype=np.float32, image_only=True,reader=ITKReader(reverse_indexing=True)),
    Resize([512,512]),
    ScaleIntensityRange( a_min=-175, a_max=250, b_min=0, b_max=1, clip=True),
    ToTensor()])
#读取上传的label文件
  
if not os.path.exists('./label_temp'):
    os.mkdir('./label_temp')
for label in uploaded_label:
    byte_label=label.read()
    ds_label=pydicom.dcmread(BytesIO(byte_label))
    
    plt.imsave("./label_temp/label_{}.png".format(label.name.split('.')[0]),ds_label.pixel_array)


#模型计算
if not os.path.exists('./mask_temp'):
    os.mkdir('./mask_temp')
if not os.path.exists('./image_temp'):
    os.mkdir('./image_temp')
model = build_resunetplusplus(1, 4).to("cpu")
checkpoint = torch.load("sarcopenia_57_0.pth",map_location=torch.device('cpu') )
model.load_state_dict(checkpoint['state_dict'])
model.eval()
with torch.no_grad():
    for image in uploaded_images:
        col1, col2,col3= st.columns(3)
        #通用名称
        image_name=image.name.split('.')[0]
        byte_img=image.read()
        ds=pydicom.dcmread(BytesIO(byte_img))
        ds.save_as(r"./image_temp/temp.dcm")
        input_tensor = preprocess_dcm(r"./image_temp/temp.dcm")
        os.remove(r"./image_temp/temp.dcm")
        #展示原始图片
        img_array = input_tensor.numpy().squeeze(0)
        img_array = img_array * 255
        img_array = img_array.astype("uint8")
        col1.write("Image_" + image.name)
        col1.image(img_array)
        #模型预测mask
        input_tensor=input_tensor.unsqueeze(0)
        output = model(input_tensor)
        output=torch.argmax(output, dim=1, keepdim=True)
        mask=output[0].squeeze(0)
        plt.imsave("./mask_temp/mask_{}.png".format(image.name.split('.')[0]),mask)
        with col2:
            st.write('Mask')
            st.image("./mask_temp/mask_{}.png".format(image.name.split('.')[0]))
        with col3:
            st.write('Label')
            st.image("./label_temp/label_{}.png".format(image.name.split('.')[0]))
if clear_button:
    #清除缓存文件夹
    for file in os.listdir('./label_temp'):
        os.remove(os.path.join('./label_temp', file))
    for file in os.listdir('./mask_temp'):
        os.remove(os.path.join('./mask_temp', file))
    for file in os.listdir('./image_temp'):
        os.remove(os.path.join('./image_temp', file))
