
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
 # streamlit_app.py

import hmac
import streamlit as st
st.set_page_config(page_title='Body Composition Measurement Tool', page_icon='📈',layout="centered", initial_sidebar_state="auto", menu_items=None)


def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


if not check_password():
    st.stop()

# Main Streamlit app starts here
st.write("Here goes your normal Streamlit app...")
st.button("Click me")
 
 
 #------------------------------------------------------
#ST用户界面 
st.header(':chart_with_upwards_trend: Consistency Testing')
st.write("**Introduction:**")
st.write("Because the model is limited by the training data, only the CT images similar to the training data can be correctly segmented and calculated, for example, similar picture quality and similar spinal cord segments, once the use of non-similar CT pictures will cause inaccurate calculation results, so it is recommended to perform a consistency test before using the APP to judge that the APP can process such data!")

"----"
st.header('Input')
col1,col2=st.columns([0.5,0.5])
with col1:
    uploaded_masks=st.file_uploader(label='Please select hand-labeled**MASK**(multiple allowed)',accept_multiple_files=True)
with col2:
    uploaded_images=st.file_uploader(label='Please select **CT** images(multiple allowed)',accept_multiple_files=True)
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

st.header("Plots and Sheets")
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
    
sata_check=st.checkbox(label='Drawing Bland-Altman plot for SATA')
if sata_check:
    data1=result_final['sata_hand']
    data2=result_final['sata_model']
    
    result = stats.ttest_rel(data1, data2)
    # 提取统计结果
    t_statistic = result.statistic
    p_value = result.pvalue
    st.write('The result of pairwise ***t*** test:')
    col1,col2=st.columns([0.5,0.5])
    col1.metric(label='t statistic', value=round(t_statistic,3))
    col2.metric(label='p value', value=round(p_value,3))
    
    f, ax = plt.subplots(1)
    sm.graphics.mean_diff_plot(data1, data2, ax = ax)
    st.pyplot(f)


sma_check=st.checkbox(label='Drawing Bland-Altman plot for SMA')
if sma_check:
    data1=result_final['sma_hand']
    data2=result_final['sma_model']
    result = stats.ttest_rel(data1, data2)
    # 提取统计结果
    t_statistic = result.statistic
    p_value = result.pvalue
    st.write('The result of pairwise ***t*** test:')
    col1,col2=st.columns([0.5,0.5])
    col1.metric(label='t statistic', value=round(t_statistic,3))
    col2.metric(label='p value', value=round(p_value,3))
    
    f, ax = plt.subplots(1)
    sm.graphics.mean_diff_plot(data1, data2, ax = ax)
    st.pyplot(f)
    
vata_check=st.checkbox(label='Drawing Bland-Altman plot for VATA')

if vata_check:
    data1=result_final['vata_hand']
    data2=result_final['vata_model']

    result = stats.ttest_rel(data1, data2)
    # 提取统计结果
    t_statistic = result.statistic
    p_value = result.pvalue
    st.write('The result of pairwise ***t*** test:')
    col1,col2=st.columns([0.5,0.5])
    col1.metric(label='t statistic', value=round(t_statistic,3))
    col2.metric(label='p value', value=round(p_value,3))
    
    f, ax = plt.subplots(1)
    sm.graphics.mean_diff_plot(data1, data2, ax = ax)
    st.pyplot(f)


