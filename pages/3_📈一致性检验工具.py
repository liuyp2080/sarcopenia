
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
#ST用户界面 
st.set_page_config(page_title='CT图像计算骨骼肌脂肪指数APP', page_icon='📈',layout="centered", initial_sidebar_state="auto", menu_items=None)
st.header(':chart_with_upwards_trend:一致性检验')
st.write("说明：")
st.write("因为模型受到训练数据的限制，仅能对与训练数据相似的CT图像进行正确的分割和计算，比如，相似的图片质量和相似的脊髓节段，一旦使用了不相似的CT图片会造成计算结果的不准确，所以推荐在使用APP之前进行一致性检验，以判断APP可以处理此类数据！")

"----"
st.header('输入区')
col1,col2=st.columns([0.5,0.5])
with col1:
    uploaded_masks=st.file_uploader(label='请选择手工标记的**MASK**（可多选）',accept_multiple_files=True)
with col2:
    uploaded_images=st.file_uploader(label='请选择原始**CT**图片（可多选）',accept_multiple_files=True)
"----"
preprocess_dcm= Compose([
    LoadImage(dtype=np.float32, image_only=True,reader=ITKReader(reverse_indexing=True)),
    # Resize([256,256]),
    ScaleIntensityRange( a_min=-175, a_max=250, b_min=0, b_max=1, clip=True),
    ToTensor()])
result={'id':[],'sata_hand':[],'vata_hand':[],'sma_hand':[]}
for mask in uploaded_masks:
    bytes_mask = mask.read()
    name_mask=mask.name.split('.')[0]
    ds=pydicom.dcmread(BytesIO(bytes_mask))
    voxel_area = ds.PixelSpacing[0] * ds.PixelSpacing[1] *(ds.Columns/ds.Rows) / 100
    mask_array=ds.pixel_array
    SATA=(mask_array==1).sum()* voxel_area
    VATA=(mask_array==2).sum()* voxel_area
    SMA = (mask_array== 3).sum() * voxel_area
    result['id'].append(name_mask)
    result['sata_hand'].append(SATA)
    result['vata_hand'].append(VATA)
    result['sma_hand'].append(SMA)
result_df=pd.DataFrame(result)

#模型计算
class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x

class Stem_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)
        # self.attn = CA_block(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        x = self.attn(x)
        project = self.c2(inputs)
        x=x+project

        return x
#main model
class ResNet_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)
        # self.attn = CA_block(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        x = self.attn(x)
        project = self.c2(inputs)
        x=x+project
        return x
        
class ASPP(nn.Module):
    def __init__(self, in_c, out_c, rate=[1, 6, 12, 18]):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[0], padding=rate[0]),
            nn.BatchNorm2d(out_c)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[1], padding=rate[1]),
            nn.BatchNorm2d(out_c)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[2], padding=rate[2]),
            nn.BatchNorm2d(out_c)
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[3], padding=rate[3]),
            nn.BatchNorm2d(out_c)
        )

        self.c5 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)


    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = x1 + x2 + x3 + x4
        y = self.c5(x)
        return y
class Attention_Block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        out_c = in_c[1]

        self.g_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(),
            nn.Conv2d(in_c[0], out_c, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2))
        )

        self.x_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(in_c[1], out_c, kernel_size=3, padding=1),
        )

        self.gc_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

    def forward(self, g, x):
        g_pool = self.g_conv(g)
        x_conv = self.x_conv(x)
        gc_sum = g_pool + x_conv
        gc_conv = self.gc_conv(gc_sum)
        y = gc_conv * x
        return y
class Decoder_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.a1 = Attention_Block(in_c)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.r1 = ResNet_Block(in_c[0]+in_c[1], out_c, stride=1)

    def forward(self, g, x):
        d = self.a1(g, x)
        d = self.up(d)
        d = torch.cat([d, g], axis=1)
        d = self.r1(d)
        return d
class build_resunetplusplus(nn.Module):
    def __init__(self,in_chanel,out_chanel):
        super().__init__()

        self.c1 = Stem_Block(in_chanel, 16, stride=1)
        self.c2 = ResNet_Block(16, 32, stride=2)
        self.c3 = ResNet_Block(32, 64, stride=2)
        self.c4 = ResNet_Block(64, 128, stride=2)

        self.b1 = ASPP(128, 256)

        self.d1 = Decoder_Block([64, 256], 128)
        self.d2 = Decoder_Block([32, 128], 64)
        self.d3 = Decoder_Block([16, 64], 32)


        self.aspp = ASPP(32, 16)
        self.output = nn.Conv2d(16, out_chanel, kernel_size=1, padding=0)

    def forward(self, inputs):
        c1 = self.c1(inputs)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)

        b1 = self.b1(c4)

        d1 = self.d1(c3, b1)
        d2 = self.d2(c2, d1)
        d3 = self.d3(c1, d2)

        output = self.aspp(d3)
        output = self.output(output)

        return output

model = build_resunetplusplus(1, 4).to("cpu")
checkpoint = torch.load("sarcopenia_57_0.pth",map_location=torch.device('cpu') )
model.load_state_dict(checkpoint['state_dict'])
model.eval()
with torch.no_grad():
    result_model={'id':[],'sata_model':[],'vata_model':[],'sma_model':[]}
    for image in uploaded_images:
        image_name=image.name.split('.')[0]
        byte_img=image.read()
        ds=pydicom.dcmread(BytesIO(byte_img))
        voxel_area = (ds.PixelSpacing[0]) * (ds.PixelSpacing[1]) *(ds.Columns/ds.Rows) / 100
        ds.save_as(r"./temp.dcm")
        input_tensor = preprocess_dcm(r"./temp.dcm")
        os.remove(r"./temp.dcm")
        input_tensor=input_tensor.unsqueeze(0)
        output = model(input_tensor)
        output=torch.argmax(output, dim=1, keepdim=True)
        SATA=(output.numpy()==1).sum()* voxel_area
        VATA=(output.numpy()==2).sum()* voxel_area
        SMA = (output.numpy() == 3).sum() * voxel_area
        result_model['id'].append(image_name)
        result_model['sata_model'].append(SATA)
        result_model['vata_model'].append(VATA)
        result_model['sma_model'].append(SMA)
result_model_df=pd.DataFrame(result_model)
result_final=pd.merge(result_df,result_model_df,on="id",how="inner")

# if not result_final.empty:
    
#     print('')

st.header("图表区")
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(result_final)
with st.container():
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='consistency_test_data.csv',
        mime='text/csv',
    )
    st.write(result_final)
    
sata_check=st.checkbox(label='绘制SATA的Bland-Altman图')
if sata_check:
    data1=result_final['sata_hand']
    data2=result_final['sata_model']

    f, ax = plt.subplots(1)
    sm.graphics.mean_diff_plot(data1, data2, ax = ax)
    st.pyplot(f)

sma_check=st.checkbox(label='绘制sma的Bland-Altman图')
if sma_check:
    data1=result_final['sma_hand']
    data2=result_final['sma_model']
    f, ax = plt.subplots(1)
    sm.graphics.mean_diff_plot(data1, data2, ax = ax)
    st.pyplot(f)
    
vata_check=st.checkbox(label='绘制vata的Bland-Altman图')

if vata_check:
    data1=result_final['vata_hand']
    data2=result_final['vata_model']

    f, ax = plt.subplots(1)
    sm.graphics.mean_diff_plot(data1, data2, ax = ax)
    st.pyplot(f)


