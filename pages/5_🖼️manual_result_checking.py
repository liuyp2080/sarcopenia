
import pydicom
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from monai.data import ITKReader
from monai.transforms import (Compose,LoadImage,Resize,ScaleIntensityRange,ToTensor)
import streamlit as st
from io import BytesIO
from sar_model import build_resunetplusplus

# streamlit_app.py

import hmac
import streamlit as st

st.set_page_config(page_title='Body Composition Measurement Tool', page_icon='🖼️',layout="centered", initial_sidebar_state="auto", menu_items=None)

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












#ST用户界面 
st.header(':frame_with_picture: Manual Result Checking')
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
