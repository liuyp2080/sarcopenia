import streamlit as st
import pandas as pd 
import numpy as np
st.set_page_config(
    page_title="Index",
    page_icon=":house:",
)

st.write("# Welcome 👋")
st.write("使用说明：")
st.write('''本APP采用人工智能深度学习模型，
         实现自动分析识别腰3段CT横断面图像内皮下脂肪、内脏脂肪和肌肉的指数，可用于判定患者多种情况下的病情及其变化。''')
col1,col2,col3,col4,col5=st.columns([0.3,0.05,0.3,0.05,0.3])
with col1:
    st.write('第一步，添加原始图片')
    st.image('demo.jpg')
with col2:
    st.write('➡️')
with col3:
    st.write("第二步，输出分割图片")
    st.image('mask_demo.png')
with col4:
    st.write('➡️')
with col5:
    st.write("第三步，展示计算结果")
    df = pd.DataFrame(
        np.random.randn(5,3),
        columns=['id','VATA','...'])
    st.table(df)
st.sidebar.success("Select a demo above.")

st.markdown(
    """
    ### 项目介绍
    
"""
)
