import time
import numpy as np
import streamlit as st
import cv2
import librosa
import librosa.display
from keras.models import load_model
import pandas as pd
import plotly.express as px
import os
from datetime import datetime
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from PIL import Image
from utils import plot_colored_polar, save_audio, get_mfccs, get_melspec, get_title
from streamlit_theme import st_theme

# load models
model = load_model("models/model3.h5")

# constants
starttime = datetime.now()

CAT6 = ["fear", "angry", "neutral", "happy", "sad", "surprise"]
CAT7 = ["fear", "disgust", "neutral", "happy", "sad", "surprise", "angry"]
CAT3 = ["positive", "neutral", "negative"]

COLOR_DICT = {
    "neutral": "grey",
    "positive": "green",
    "happy": "green",
    "surprise": "orange",
    "fear": "purple",
    "negative": "red",
    "angry": "red",
    "sad": "lightblue",
    "disgust": "brown",
}

TEST_CAT = ["fear", "disgust", "neutral", "happy", "sad", "surprise", "angry"]
TEST_PRED = np.array([0.3, 0.3, 0.4, 0.1, 0.6, 0.9, 0.1])

st.set_page_config(page_title="SER", page_icon=":speech_balloon:", layout="wide")

theme = st_theme()


def color_dict(coldict=COLOR_DICT):
    return COLOR_DICT


st.sidebar.subheader("Menu")
website_menu = st.sidebar.selectbox(
    "Menu",
    (
        "Emotion Recognition",
        "Project description",
        "Our team",
    ),
)

if website_menu == "Emotion Recognition":
    em3 = em6 = em7 = gender = False
    st.sidebar.subheader("Settings")

    st.markdown("## Upload the file")
    audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg"])
    if audio_file is not None:
        if not os.path.exists("audio"):
            os.makedirs("audio")
        path = os.path.join("audio", audio_file.name)
        if_save_audio = save_audio(audio_file)
        if if_save_audio == 1:
            st.warning("File size is too large. Try another file.")
        elif if_save_audio == 0:
            st.audio(audio_file, format="audio/wav", start_time=0)
            try:
                wav, sr = librosa.load(path, sr=44100)
                Xdb = get_melspec(path)[1]
                mfccs = librosa.feature.mfcc(y=wav, sr=sr)
            except Exception as e:
                audio_file = None
                st.error(
                    f"Error {e} - wrong format of the file. Try another .wav file."
                )
        else:
            st.error("Unknown error")
    else:
        if st.button("Try test file"):
            wav, sr = librosa.load("test.wav", sr=44100)
            Xdb = get_melspec("test.wav")[1]
            mfccs = librosa.feature.mfcc(y=wav, sr=sr)
            st.audio("test.wav", format="audio/wav", start_time=0)
            path = "test.wav"
            audio_file = "test"

    em3 = st.sidebar.checkbox("3 emotions", True)
    em6 = st.sidebar.checkbox("6 emotions", True)
    em7 = st.sidebar.checkbox("7 emotions")
    gender = st.sidebar.checkbox("gender")

    # with st.sidebar.expander("Change colors"):
    #     num_cols = 3
    #     md = """
    #     <p style='color: grey; padding-top: 30px;'>
    #         Use these options after you've got the plots
    #     </p>
    #     """
    #     st.sidebar.markdown(md, unsafe_allow_html=True)
    #     col_list = [st.columns(num_cols) for _ in range(num_cols)]
    #     color_names = [
    #         "Angry",
    #         "Fear",
    #         "Disgust",
    #         "Sad",
    #         "Neutral",
    #         "Surprise",
    #         "Happy",
    #         "Negative",
    #         "Positive",
    #     ]
    #     default_colors = [
    #         "#FF0000",
    #         "#800080",
    #         "#A52A2A",
    #         "#ADD8E6",
    #         "#808080",
    #         "#FFA500",
    #         "#008000",
    #         "#FF0000",
    #         "#008000",
    #     ]
    #     color_inputs = []

    #     for idx, col in enumerate(col_list):
    #         for i, c in enumerate(col):
    #             color_input = c.color_picker(
    #                 color_names[i * num_cols + idx],
    #                 value=default_colors[i * num_cols + idx],
    #             )
    #             color_inputs.append(color_input)

    #     if st.button("Update colors"):
    #         global COLOR_DICT
    #         COLOR_DICT = {
    #             "angry": color_inputs[0],
    #             "fear": color_inputs[1],
    #             "disgust": color_inputs[2],
    #             "sad": color_inputs[3],
    #             "neutral": color_inputs[4],
    #             "surprise": color_inputs[5],
    #             "happy": color_inputs[6],
    #             "negative": color_inputs[7],
    #             "positive": color_inputs[8],
    #         }
    #         st.success(COLOR_DICT)

    if audio_file is not None:
        st.markdown("## Analyzing...")
        if not audio_file == "test":
            st.sidebar.subheader("Audio file")
            file_details = {
                "Filename": audio_file.name,
                "FileSize": audio_file.size,
            }
            st.sidebar.write(file_details)

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                fig = plt.figure(figsize=(10, 2))
                fig.set_facecolor(theme["backgroundColor"])
                plt.title("MFCCs", color=theme["textColor"])
                librosa.display.specshow(mfccs, sr=sr, x_axis="time")
                plt.gca().get_yaxis().set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                plt.gca().spines["left"].set_visible(False)
                plt.gca().spines["top"].set_visible(False)
                st.write(fig)
            with col2:
                fig2 = plt.figure(figsize=(10, 2))
                fig2.set_facecolor(theme["backgroundColor"])
                plt.title("Mel-log-spectrogram", color=theme["textColor"])
                librosa.display.specshow(Xdb, sr=sr, x_axis="time", y_axis="hz")
                plt.gca().get_yaxis().set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                plt.gca().spines["left"].set_visible(False)
                plt.gca().spines["top"].set_visible(False)
                st.write(fig2)

        st.markdown("## Predictions")
        with st.container():
            col1, col2 = st.columns(2)
            mfccs = get_mfccs(path, model.input_shape[-1])
            mfccs = mfccs.reshape(1, *mfccs.shape)
            pred = model.predict(mfccs)[0]
            with col1:
                if em3:
                    pos = pred[3] + pred[5] * 0.5
                    neu = pred[2] + pred[5] * 0.5 + pred[4] * 0.5
                    neg = pred[0] + pred[1] + pred[4] * 0.5
                    data3 = np.array([pos, neu, neg])
                    txt = "MFCCs\n" + get_title(data3, CAT3)
                    fig = plt.figure(figsize=(5, 5))
                    COLORS = color_dict(COLOR_DICT)
                    plot_colored_polar(
                        fig,
                        predictions=data3,
                        categories=CAT3,
                        title=txt,
                        colors=COLORS,
                        facecolor=theme["backgroundColor"],
                        fontcolor=theme["textColor"],
                    )
                    st.write(fig)
            with col2:
                if em6:
                    txt = "MFCCs\n" + get_title(pred, CAT6)
                    fig2 = plt.figure(figsize=(5, 5))
                    COLORS = color_dict(COLOR_DICT)
                    plot_colored_polar(
                        fig2,
                        predictions=pred,
                        categories=CAT6,
                        title=txt,
                        colors=COLORS,
                        facecolor=theme["backgroundColor"],
                        fontcolor=theme["textColor"],
                    )
                    st.write(fig2)

            col1, col2 = st.columns(2)
            with col1:
                if em7:
                    model_ = load_model("models/model4.h5")
                    mfccs_ = get_mfccs(path, model_.input_shape[-2])
                    mfccs_ = mfccs_.T.reshape(1, *mfccs_.T.shape)
                    pred_ = model_.predict(mfccs_)[0]
                    txt = "MFCCs\n" + get_title(pred_, CAT7)
                    fig3 = plt.figure(figsize=(5, 5))
                    COLORS = color_dict(COLOR_DICT)
                    plot_colored_polar(
                        fig3,
                        predictions=pred_,
                        categories=CAT7,
                        title=txt,
                        colors=COLORS,
                        facecolor=theme["backgroundColor"],
                        fontcolor=theme["textColor"],
                    )
                    st.write(fig3)
            with col2:
                if gender:
                    with st.spinner("Wait for it..."):
                        gmodel = load_model("models/model_mw.h5")
                        gmfccs = get_mfccs(path, gmodel.input_shape[-1])
                        gmfccs = gmfccs.reshape(1, *gmfccs.shape)
                        gpred = gmodel.predict(gmfccs)[0]
                        gdict = [["female", "woman.png"], ["male", "man.png"]]
                        ind = gpred.argmax()
                        txt = "Predicted gender: " + gdict[ind][0]
                        img = Image.open("images/" + gdict[ind][1])

                        fig4 = plt.figure(figsize=(3, 3))
                        fig4.set_facecolor(theme["backgroundColor"])
                        plt.title(txt, color=theme["textColor"], fontsize=8)
                        plt.imshow(img)
                        plt.axis("off")
                        st.write(fig4)

elif website_menu == "Project description":
    st.title("Project description")
    st.subheader("GitHub")
    link = "[GitHub repository of the web-application]" "(https://github.com/rnxj/ser)"
    st.markdown(link, unsafe_allow_html=True)

    st.subheader("Theory")
    link = (
        "[Theory behind - Medium article]"
        "(https://talbaram3192.medium.com/classifying-emotions-using-audio-recordings-and-python-434e748a95eb)"
    )
    with st.expander("See Wikipedia definition"):
        components.iframe(
            "https://en.wikipedia.org/wiki/Emotion_recognition",
            height=320,
            scrolling=True,
        )

    st.subheader("Dataset")
    txt = """
        This web-application is a part of the final **Data Mining** project for **ITC Fellow Program 2020**. 

        Datasets used in this project
        * Crowd-sourced Emotional Mutimodal Actors Dataset (**Crema-D**)
        * Ryerson Audio-Visual Database of Emotional Speech and Song (**Ravdess**)
        * Surrey Audio-Visual Expressed Emotion (**Savee**)
        * Toronto emotional speech set (**Tess**)    
        """
    st.markdown(txt, unsafe_allow_html=True)

    df = pd.read_csv("df_audio.csv")
    fig = px.violin(
        df,
        y="source",
        x="emotion4",
        color="actors",
        box=True,
        points="all",
        hover_data=df.columns,
    )
    st.plotly_chart(fig, use_container_width=True)

elif website_menu == "Our team":
    st.subheader("Our team")
    st.balloons()
    st.info("Reuel Nixon - 21BPS1406")
    st.info("Alapati Hemalatha - 21BPS1384")
