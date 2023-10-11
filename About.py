import streamlit as st
import pandas as pd 
import numpy as np
st.set_page_config(
    page_title="Index",
    page_icon=":house:",
)

st.success("# Welcome 👋")
st.info('''
         **APP Introduction**\n
         This app was designed to help clinicians and researchers better identify and quantify changes in body fat and muscle.\n
         Three application modes are provided:\n
         1. Population analysis: this mode suites for analyzing CT images of different patients.\n
         2. Time series analysis: this mode suites for analyzing CT images of the same patient at a time series.\n
         3. Consistency checking: checking whether new dataset suitable for the application.\n
         ''')
st.write('The APP can easily used with three steps:')
col1,col2,col3,col4,col5=st.columns([0.3,0.05,0.3,0.05,0.3])
with col1:
    st.write('First step,input CT images')
    st.image('demo.jpg')
with col2:
    st.write('➡️')
with col3:
    st.write("Second step,output mask")
    st.image('mask_demo.png')
with col4:
    st.write('➡️')
with col5:
    st.write("Third step，demonstrate the result")
    df = pd.DataFrame(
        np.random.randn(5,3),
        columns=['id','VATA','...'])
    st.table(df)

st.markdown(
    """
<h2 style='color: #ff0000;'>ABOUT</h2>
<p style="text-align: justify;background-color: #f2f2f2">
With the rapid rise of deep learning technology, image segmentation is playing an increasingly important role in the field of medical imaging. Especially in the processing and analysis of CT (computed tomography) images, deep learning algorithms provide researchers and doctors with powerful tools to explore changes in body composition.</p>
<p style="text-align: justify;background-color: #f2f2f2">CT is a non-invasive medical imaging technique that can visualize the internal structures of the human body in a cross-sectional form, including bones, organs, blood vessels, etc., however, it is a challenging task to accurately segment specific structures from complex CT images. While traditional rule-based and manual feature-based methods do not work well in complex situations, deep learning automatically learns features and patterns through neural networks, becoming an effective image segmentation method. </p>
<p style="text-align: justify;background-color: #f2f2f2">The advent of deep learning has brought significant improvements to CT image segmentation. Convolutional neural networks (CNNs) are one of the most commonly used deep learning architectures, efficiently extracting features from images through multi-layer convolution and pooling operations. In the image segmentation task, U-Net is a common CNN structure, which adopts the structure of encoder-decoder, which can realize pixel-level segmentation. By using a large amount of labeled CT image data during the training process, the deep learning algorithm can learn the features of different tissues and structures and accurately segment the region of interest. </p>
<p style="text-align: justify;background-color: #f2f2f2">Using deep learning for CT image segmentation to identify body composition, researchers and doctors can better understand changes in body composition. For example, when observing CT images of patients over time, deep learning algorithms can help automatically identify and quantify changes in body fat, muscle, and bone, so as to better understand the development of body composition. In addition, deep learning can accurately segment structures such as tumors, organs, and blood vessels, helping doctors detect lesions early and evaluate treatment effects.</p>
<p style="text-align: justify;background-color: #f2f2f2">Our team is committed to the development of clinical translational application of artificial intelligence, in 2022, the team used AI for the first time in sarcopenia research to assist research, after half a year of experimental running-in, we finally used deep learning methods to successfully build an AI-based CT image segmentation model, which is used to segment the muscle and fat composition in the image, and output the corresponding skeletal muscle index and fat index, which is convenient for clinicians to quantitatively evaluate the patient's systemic condition.</p>
<p style="text-align: justify;background-color: #f2f2f2">With the further increase of the aging population, the incidence of sarcopenia worldwide is also increasing year by year, and the judgment of sarcopenia requires a quantitative standard, of which the muscle area measured by lumbar 3-level CT is the currently recognized quantitative standard. However, there is a cost of time and manpower in labor-based measurement, and the corresponding AI assistance is more convenient and efficient. Based on this idea, our team has now successfully developed the "Body Composition Measurement Tool 1.0" based on CT as an AI-assisted mapping tool ---- CT. Help clinicians and researchers better identify and quantify changes in body fat, muscle and bone, understand the development and changes of body composition, and assist clinical and scientific research.</p>

""",unsafe_allow_html=True
)
