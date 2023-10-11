import streamlit as st
import os
import torch.nn as nn
import torch
from monai.data import ITKReader
from monai.transforms import (Compose,LoadImage,Resize,ScaleIntensityRange,ToTensor)
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
st.set_page_config(page_title='Body Composition Measurement Tool', page_icon='👫', layout="centered", initial_sidebar_state="auto", menu_items=None)
st.header('👫 Population Analysis')
st.write('''
**Instructions:**\n
Just support ***dicom*** format CT images.\n
This module is suitable for analyzing CT images of different patients at the same time point, obtaining the fat and muscle indices of patients, and forming a table to provide reference for understanding the changes of fat muscle index in similar patient populations under various circumstances
**The Substractions:** \n
SATA: subcutaneous adipose tissue area; VATA: visceral adipose tissue area; MSA: muscle area
''')

"----"
st.header('Input Area')
col1,col2=st.columns([0.8,0.2])
with col1:
    uploaded_files=st.file_uploader(label='Please select CT images(multiple allowed)',accept_multiple_files=True)

preprocess= Compose([
    ScaleIntensityRange( a_min=0, a_max=256, b_min=0, b_max=1, clip=True),
    ToTensor()])

preprocess_dcm= Compose([
    LoadImage(dtype=np.float32, image_only=True,reader=ITKReader(reverse_indexing=True)),
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
            voxel_area = ds.PixelSpacing[0] * ds.PixelSpacing[1] *(ds.Columns/ds.Rows) / 100
        # print(ds.PixelSpacing[0], ds.PixelSpacing[1])
        except:
            voxel_area = 999
            if not hasattr(ds, 'PixelSpacing'):
                st.error("Error: PixelSpacing not found. The result may not be correct.")
        ds.save_as(r"./temp.dcm")

        image = preprocess_dcm(r"./temp.dcm")
        os.remove(r"./temp.dcm")
        img_array = image.numpy().squeeze(0)
        img_array = img_array * 255
        img_array = img_array.astype("uint8")
        col1.write("Image_" + uploaded_file.name)
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
            raise ValueError('Error: Please select ***dicom*** format CT images.')
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
            results[f"h_{i}"]=st.number_input('height(m)：',0.00,3.00,value=1.7,key=f"h_{i}")
            # if voxel_area == 0:
            #     voxel_area = pixel_h * pixel_w*4/100
            #     results[f'pixel_h_{i}']=pixel_h
            #     results[f'pixel_w_{i}']=pixel_w
            results[f"voxel_{i}"]=voxel_area
        with col2:
            SATA=(mask.numpy()==1).sum()*results[f"voxel_{i}"]
            VATA=(mask.numpy()==2).sum()*results[f"voxel_{i}"]
            SMA = (mask.numpy() == 3).sum() * results[f"voxel_{i}"]
            SATI=SATA/(results[f"h_{i}"]*results[f"h_{i}"])
            VATI=VATA/(results[f"h_{i}"]*results[f"h_{i}"])
            SMI=SMA/(results[f"h_{i}"]*results[f"h_{i}"])

            results[f'SATA_{i}']=SATA
            results[f"VATA_{i}"]=VATA
            results[f'SMA_{i}']=SMA
            results[f"SATI_{i}"]=SATI
            results[f"VATI_{i}"]=VATI
            results[f'SMI_{i}']=SMI

            st.latex(f"SATA,cm^2:{'{:.3f}'.format(results[f'SATA_{i}'])}")
            st.latex(f"VATA,cm^2:{'{:.3f}'.format(results[f'VATA_{i}'])}")
            st.latex(f"SMA,cm^2:{'{:.3f}'.format(results[f'SMA_{i}'])}")
            st.latex(f"SATI,cm^2/m:{'{:.3f}'.format(results[f'SATI_{i}'])}")
            st.latex(f"VATI,cm^2/m:{'{:.3f}'.format(results[f'VATI_{i}'])}")
            st.latex(f"SMI,cm^2/m:{'{:.3f}'.format(results[f'SMI_{i}'])}")
    num+=1
    file_name.append(uploaded_file.name.split('.')[0])
#
id= file_name   
height=[results[f'h_{i}'] for i in range(num)]
SATA=[results[f'SATA_{i}'] for i in range(num)]
VATA=[results[f"VATA_{i}"] for i in range(num)]
SMA=[results[f'SMA_{i}'] for i in range(num)]
SATI=[results[f"SATI_{i}"] for i in range(num)]
VATI=[results[f"VATI_{i}"] for i in range(num)]
SMI=[results[f'SMI_{i}'] for i in range(num)]
df_results=pd.DataFrame({'id':id,'height':height,'SATA':SATA,'VATA':VATA,'SMA':SMA,'SATI':SATI,'VATI':VATI,'SMI':SMI})
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df_results)
'-----'
import plotly.express as px
st.header('Plots and Sheets')
with st.container():
    plot_results=pd.melt(df_results,id_vars=['id'],value_vars=['SATA','VATA','SMA','SATI','VATI','SMI'],var_name='variables',value_name='value')
    fig=px.box(plot_results,x='variables',y='value',color='variables',points='all')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig,use_container_width=True)
        
with st.container():
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='results.csv',
        mime='text/csv',
    )
    st.dataframe(df_results,use_container_width=False)


