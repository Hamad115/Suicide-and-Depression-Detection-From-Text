import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.title("Depression Detection Using Text")

image = Image.open('depression image.png')
st.image(image, use_column_width = True)

st.write(
    """
    ## Enter some text that will be classified as Suicide, Depression or something that a Teenager would write.
    ## The dataset is a collection of posts from "SuicideWatch" and "depression" subreddits of the Reddit platform.
    
    ***
    """
)

st.subheader("Enter some text to be classified")

input_text = st.text_area("Text: ", height =300)

from model import transform_predict

prediction, prediction_proba = transform_predict(input_text)

st.subheader('Class labels and their corresponding Index number')
dataframe = pd.DataFrame(["SuicideWatch", "depression", "teenagers"])
st.write(dataframe)

# Displaying the results
st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

# Prediction Probability Graph
import streamlit.elements.plotly_chart
import matplotlib.pyplot
import plotly.express as px

prediction_probability = pd.DataFrame(prediction_proba)

st.subheader('Prediction Probability Bar Graph')
st.bar_chart(prediction_proba)