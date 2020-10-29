import os
import streamlit as st 
from deep_cnn import DeepModel
from PIL import Image
import img_process
import matplotlib.pyplot as plt 
import math 
import numpy as np
import torch
import io
import pickle
import pandas as pd
# main applications of the project
class VTCNNApplication(object):
    def __init__(self):
        self.deep_model = DeepModel("ResNet-18")
        self.cache_root = "./download_cache/"
        if not os.path.isdir(self.cache_root):
            os.makedirs(self.cache_root)
        
        # self.default_image = "./src/mnist_test.jpg"
        st.set_page_config(page_title="VTDNN", page_icon="./src/vtcnn_16x16.ico")
        self.default_image = "./src/cat_01.jpg"

    def loadImage(self, path, img_width=300, img_height=300):
        img_data = Image.open(path)
        img_data = img_data.convert("RGB")
        img_data_resize = img_data.resize((img_width, img_height), Image.ANTIALIAS)
        return img_data, img_data_resize

    def runApp(self):
        # define page 
        st.sidebar.markdown("""
        ## Welcome to use VTDNN
        > by iceory

        > since 2020.10.29

        **This project is based on Streamlit.**
        """)

        # set page title and icon

        # get model name
        model_name = st.sidebar.selectbox("Please choose a model: ", self.deep_model.model_list, index=1)
        self.deep_model = DeepModel(model_name)
        # view_img_col, view_weight_col = st.beta_columns(2)
        # with view_weight_col:
        #     model_weight = st.file_uploader("Please upload weights (*.pkl): (optional)", accept_multiple_files=False, type=["pth", "pkl"])
        #     weight_file_path = None
        #     if model_weight is not None:
        #         st.write("filename is: {}".format(model_weight.name))
        #         weight_file_path = os.path.join(self.cache_root, model_weight.name)
        #         bytes_data = io.BytesIO(model_weight.getvalue())
        #         state_dict = torch.load(bytes_data)
        #         print(state_dict)
        #         torch.save(state_dict, weight_file_path)
 
        # get test input
        # with view_img_col:
        upload_input = st.file_uploader("Please upload your test image: (optional)", accept_multiple_files=False, type=["png", "jpeg", "jpg"])
        img_file_path = None
        if upload_input is not None:
            img_file_path = os.path.join(self.cache_root, upload_input.name)
            image = Image.open(upload_input)
            image.save(img_file_path)
            self.default_image = img_file_path


        layer_list = list(self.deep_model.layer_feature.keys())
        select_layer = st.sidebar.selectbox("Please choose a layer: ", layer_list)

        # config
        softmax_flag = st.sidebar.checkbox("Softmax Prediction")
        # load test input
        test_input, test_input_resize = self.loadImage(self.default_image)
        st.sidebar.markdown("### Input image: ")
        st.sidebar.image(test_input_resize)
        st.markdown("### Prediction: ")
        msg_str = self.deep_model.forward(test_input)
        st.info(msg_str)

        # show features
        select_feature = self.deep_model.layer_feature[select_layer]
        st.markdown("### Feature Visualization of {}".format(select_layer))
        if select_feature.ndim == 2:
            st.markdown("*Note: feature size is {}, reduced to bar_chart*".format(select_feature.shape))
            if softmax_flag:
                select_feature = torch.nn.functional.softmax(select_feature, dim=1)
            
            st.bar_chart(select_feature.detach().squeeze().cpu().numpy())

        elif select_feature.ndim == 4:
            npimg = img_process.tensor2Grid(select_feature)
            img_data_convert = Image.fromarray(plt.cm.jet(npimg[...,0], bytes=True))
            st.markdown("*Note: feature size is {}*".format(select_feature.shape))
            st.image(img_data_convert, use_column_width=True)
        else:
            st.error("unsupported feature shape: {}".format(select_feature.size()))
        

if __name__ == "__main__":
    app = VTCNNApplication()
    app.runApp()