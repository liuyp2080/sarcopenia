import streamlit as st
import pandas as pd 
import numpy as np
from PIL import Image
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
<p style="text-al ign: justify">
With the rapid rise of deep learning technology, image segmentation is playing an increasingly important role in the field of medical imaging. Especially in the processing and analysis of CT (computed tomography) images, deep learning algorithms provide researchers and doctors with powerful tools to explore changes in body composition.</p>
<p style="text-align: justify">CT is a non-invasive medical imaging technique that can visualize the internal structures of the human body in a cross-sectional form, including bones, organs, blood vessels, etc., however, it is a challenging task to accurately segment specific structures from complex CT images. While traditional rule-based and manual feature-based methods do not work well in complex situations, deep learning automatically learns features and patterns through neural networks, becoming an effective image segmentation method. </p>
<p style="text-align: justify">The advent of deep learning has brought significant improvements to CT image segmentation. Convolutional neural networks (CNNs) are one of the most commonly used deep learning architectures, efficiently extracting features from images through multi-layer convolution and pooling operations. In the image segmentation task, U-Net is a common CNN structure, which adopts the structure of encoder-decoder, which can realize pixel-level segmentation. By using a large amount of labeled CT image data during the training process, the deep learning algorithm can learn the features of different tissues and structures and accurately segment the region of interest. </p>
<p style="text-align: justify">Using deep learning for CT image segmentation to identify body composition, researchers and doctors can better understand changes in body composition. For example, when observing CT images of patients over time, deep learning algorithms can help automatically identify and quantify changes in body fat, muscle, and bone, so as to better understand the development of body composition. In addition, deep learning can accurately segment structures such as tumors, organs, and blood vessels, helping doctors detect lesions early and evaluate treatment effects.</p>
<p style="text-align: justify">Our team is committed to the development of clinical translational application of artificial intelligence, in 2022, the team used AI for the first time in sarcopenia research to assist research, after half a year of experimental running-in, we finally used deep learning methods to successfully build an AI-based CT image segmentation model, which is used to segment the muscle and fat composition in the image, and output the corresponding skeletal muscle index and fat index, which is convenient for clinicians to quantitatively evaluate the patient's systemic condition.</p>
<h2 style='color: #ff0000;'>Model Efficiency</h2>
<p style="text-align: justify">The efficacy of different models for segmenting subcutaneous adipose tissue (SATA), visceral adipose tissue (VATA), and skeletal muscle (SMA) on CT images was evaluated using a comprehensive set of metrics, including Dice coefficient, Jaccard index, Hausdorff distance (HD95), precision, and recall. The prediction accuracies are shown in Table 2 for the L3BCSM models, the UNETR models and AHNET models. L3BCSM model achieved high scores on all metrics, comparable to UNETR and AHNET models. UNETR model had slightly higher Dice coefficient and Jaccard index than L3BCSM and AHNET models, but slightly higher Hausdorff distance than L3BCSM model. AHNET model had slightly lower Dice coefficient and Jaccard index than UNETR and L3BCSM models, but slightly lower Hausdorff distance than UNETR model. Overall, the L3BCSM, UNETR, and AHNET models are all effective models for body composition image segmentation. The L3BCSM model has high accuracy and robustness, making it a potential candidate for clinical applications(table2).</p>
<p style="text-align: justify"></p>
""",unsafe_allow_html=True
)
img_table1=Image.open('table1.png')
img_table2=Image.open('table2.png')
img_table3=Image.open('table3.png')
st.image(img_table1)
st.image(img_table2)

st.markdown(
    """
    <p style="text-align: justify">Cohort 2 is a dataset provided by the General Hospital of the Northern Theater of Operations, which two radiologists with senior professional titles manually annotate to ensure the high quality of manual annotation. The generalization ability and objectivity of the L3BCSM model were evaluated on the external dataset(cohort 2). The results showed that L3BCSM achieved satisfied performance on the dataset(Table 3). </p>
    """
,unsafe_allow_html=True)
st.image(img_table3)
