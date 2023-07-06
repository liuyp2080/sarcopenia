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
checkpoint = torch.load("sarcopenia.pth",map_location=torch.device('cpu') )
model.load_state_dict(checkpoint['state_dict'])
model.eval()

#ST用户界面 
st.set_page_config(page_title='CT图像计算骨骼肌脂肪指数APP', page_icon='⏲️',layout="centered", initial_sidebar_state="auto", menu_items=None)
st.header(':timer_clock:系列时间点分析')
st.write("说明：")
st.write("建议使用dicom格式的图像，如上传非Dicom格式图片，请确保CT窗口为[-175,250]")
st.write("本APP适用于分析同一个患者不同时间点的CT图像，获取不同时间点的脂肪/肌肉指数，并绘制趋势图，为多种情况下评估患者病情随时间的变化提供参考。")

"----"
st.header('输入区')
col1,col2=st.columns([0.6,0.4])
with col1:
    uploaded_files=st.file_uploader(label='请选择CT图片（可多选）',accept_multiple_files=True)
with col2:
    height=st.number_input('患者身高(m)：',0.00,3.00,value=1.7)
    st.write('非dicomCT图像，请输入：')
    pixel_h = st.number_input('体素_H(mm)：', 0.7)
    pixel_w = st.number_input('体素_W(mm)：', 0.7)

preprocess= Compose([
    Resize([256,256]),
    ScaleIntensityRange( a_min=0, a_max=256, b_min=0, b_max=1, clip=True),
    ToTensor()])

preprocess_dcm= Compose([
    LoadImage(dtype=np.float32, image_only=True,reader=ITKReader(reverse_indexing=True)),
    Resize([256,256]),
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
            voxel_area = ds.PixelSpacing[0] * ds.PixelSpacing[1] * 4 / 100
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
            col1.write("原始图片_"+uploaded_file.name)
            col1.image(img_array,clamp=True)
        except:
            raise ValueError('图片格式错误')
    #预测
    with torch.no_grad():
        mask = model(image.unsqueeze(0))
        mask = torch.argmax(mask, dim=1, keepdim=True)
        mask=mask[0].squeeze(0)
        if not os.path.exists('./mask'):
            os.mkdir('./mask')
        plt.imsave("./mask/mask_{}.png".format(uploaded_file.name.split('.')[0]),mask)

    with col2:
        st.write('分割图片')
        st.image("./mask/mask_{}.png".format(uploaded_file.name.split('.')[0]))
    #input related data and get result    
    with st.expander('修改患者信息，计算相关参数',expanded=False):
        col1,col2=st.columns([0.5,0.5])#[]is the ratio of two columns
        with col1:
            results[f'period_{i}']=st.number_input('随访时间',value=0,key=f'period_{i}')
            if voxel_area == 0:
                voxel_area = pixel_h * pixel_w*4/100
                results[f'pixel_h_{i}']=pixel_h
                results[f'pixel_w_{i}']=pixel_w
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
st.header('图表区')
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
