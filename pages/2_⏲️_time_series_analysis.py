import streamlit as st
import os
import torch.nn as nn
import torch
from monai.data import ITKReader,PILReader
from monai.transforms import (Compose,LoadImage,Resize,EnsureChannelFirst,ScaleIntensityRange,ToTensor,Rotate)
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import pandas as pd
from sar_model import build_resunetplusplus


model = build_resunetplusplus(1, 4).to("cpu")
checkpoint = torch.load("sarcopenia_57_0.pth",map_location=torch.device('cpu') )
model.load_state_dict(checkpoint['state_dict'])
model.eval()

#ST用户界面 
st.set_page_config(page_title='Body Composition Measurement Tool', page_icon='⏲️',layout="centered", initial_sidebar_state="auto", menu_items=None)
st.header(':timer_clock: Time Series Analysis')
st.write("Ins**tructions:**")
st.write("Just support ***dicom*** format CT images.")
st.write("This module is suitable for analyzing CT images of the same patient at different time points, obtaining fat/muscle index at different time points, and plotting trend maps, which provides reference for evaluating the change of the patient's condition over time in a variety of situations.")

"----"
st.header('Input Area')
col1,col2=st.columns([0.6,0.4])
with col1:
    uploaded_files=st.file_uploader(label='Please select CT images(multiple allowed)',accept_multiple_files=True)
with col2:
    height=st.number_input('the height(m)：',0.00,3.00,value=1.7)

preprocess= Compose([
    # Resize([256,256]),
    ScaleIntensityRange( a_min=0, a_max=256, b_min=0, b_max=1, clip=True),
    ToTensor()])

preprocess_dcm= Compose([
    LoadImage(dtype=np.float32, image_only=True,reader=ITKReader(reverse_indexing=True)),
    # Resize([256,256]),
    ScaleIntensityRange( a_min=-175, a_max=250, b_min=0, b_max=1, clip=True),
    ToTensor()])

from io import BytesIO
from PIL import Image
import itk
results={}
num=0
file_name=[]
for i,uploaded_file in enumerate(uploaded_files):
    "----"
    voxel_area=0
    col1, col2= st.columns(2)
    #转成灰度图片并展示
    bytes_data = uploaded_file.read()
    file_format = uploaded_file.name.split('.')[-1]

    if file_format =='dcm':
        ds=pydicom.dcmread(BytesIO(bytes_data))
        try:
            voxel_area = ds.PixelSpacing[0] * ds.PixelSpacing[1] * (ds.Columns/ds.Rows) / 100
        except:
            voxel_area = 999
            if not hasattr(ds, 'PixelSpacing'):
                st.error("Error: PixelSpacing not found. The result may not be correct.")
        # print(ds.pixel_array.max(),ds.pixel_array.min())
        ds.save_as(r"./temp.dcm")

        image = preprocess_dcm(r"./temp.dcm")
        os.remove(r"./temp.dcm")
        img_array = image.numpy().squeeze(0)
        img_array = img_array * 255
        img_array = img_array.astype("uint8")
        col1.write("原始图片_" + uploaded_file.name)
        col1.image(img_array)

    else:
        try:
            image = Image.open(BytesIO(bytes_data)).convert('L')

            image = torch.tensor(np.array(image).astype("int16")).unsqueeze(0)
            image=preprocess(image)
            # image=image.transpose(2,1)
            img_array=image.numpy().squeeze(0)
            img_array=img_array*255
            img_array = img_array.astype("uint8")
            col1.write("Image_"+uploaded_file.name)
            col1.image(img_array,clamp=True)
        except:
            raise ValueError('wrong image format')
    #预测
    with torch.no_grad():
        mask = model(image.unsqueeze(0))
        mask = torch.argmax(mask, dim=1, keepdim=True)
        mask=mask[0].squeeze(0)
        if not os.path.exists('./mask'):
            os.mkdir('./mask')
        plt.imsave("./mask/mask_{}.png".format(uploaded_file.name.split('.')[0]),mask)

    with col2:
        st.write('Mask')
        st.image("./mask/mask_{}.png".format(uploaded_file.name.split('.')[0]))
    #input related data and get result    
    with st.expander('Fulfill the Info of Patients',expanded=False):
        col1,col2=st.columns([0.5,0.5])#[]is the ratio of two columns
        with col1:
            results[f'period_{i}']=st.number_input('Timepoints',value=0,key=f'period_{i}')
            # if voxel_area == 0:
            #     voxel_area = pixel_h * pixel_w*4/100
            #     results[f'pixel_h_{i}']=pixel_h
            #     results[f'pixel_w_{i}']=pixel_w
            results[f"voxel_{i}"]=voxel_area


        with col2:
            SATA=(mask.numpy()==1).sum()*results[f"voxel_{i}"]
            VATA=(mask.numpy()==2).sum()*results[f"voxel_{i}"]
            SMA = (mask.numpy() == 3).sum() * results[f"voxel_{i}"]
            SATI=SATA/(height*height)
            VATI=VATA/(height*height)
            SMI=SMA/(height*height)

            results[f'SATA_{i}']=SATA
            results[f"VATA_{i}"]=VATA
            results[f'SMA_{i}']=SMA
            results[f"SATI_{i}"]=SATI
            results[f"VATI_{i}"]=VATI
            results[f'SMI_{i}']=SMI

            st.latex(f"皮下脂肪面积(SATA,cm^2)：{'{:.3f}'.format(results[f'SATA_{i}'])}")
            st.latex(f"内脏脂肪面积(VATA,cm^2)：{'{:.3f}'.format(results[f'VATA_{i}'])}")
            st.latex(f"骨骼肌面积(SMA,cm^2)：{'{:.3f}'.format(results[f'SMA_{i}'])}")
            st.latex(f"皮下脂肪指数(SATI,cm^2/m)：{'{:.3f}'.format(results[f'SATI_{i}'])}")
            st.latex(f"内脏脂肪指数(VATI,cm^2/m))：{'{:.3f}'.format(results[f'VATI_{i}'])}")
            st.latex(f"骨骼肌指数(SMI,cm^2/m))：{'{:.3f}'.format(results[f'SMI_{i}'])}")
    num+=1
    file_name.append(uploaded_file.name.split('.')[0])
#
id= file_name
period=[results[f'period_{i}'] for i in range(num)]   
heights=np.repeat(height,num)
SATA=[results[f'SATA_{i}'] for i in range(num)]
VATA=[results[f"VATA_{i}"] for i in range(num)]
SMA=[results[f'SMA_{i}'] for i in range(num)]
SATI=[results[f"SATI_{i}"] for i in range(num)]
VATI=[results[f"VATI_{i}"] for i in range(num)]
SMI=[results[f'SMI_{i}'] for i in range(num)]
df_results=pd.DataFrame({'id':id,'period':period,'height':heights,'SATA':SATA,'VATA':VATA,'SMA':SMA,'SATI':SATI,'VATI':VATI,'SMI':SMI})
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df_results)

'---'
st.header('Plots and Sheets')
with st.container():
    col1,col2=st.columns(2)
    with col1:
        st.line_chart(data=df_results,y=['SATA','VATA','SMA'],x='period',use_container_width=True)
    with col2:
        st.line_chart(data=df_results,y=['SATI','VATI','SMI'],x='period',use_container_width=True)
'-----'
with st.container():
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='results.csv',
        mime='text/csv',
    )
    st.dataframe(df_results,use_container_width=True)
