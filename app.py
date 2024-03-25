import numpy as np
import streamlit as st
import librosa
import librosa.display
import pandas as pd
import plotly.express as px
import os
import streamlit.components.v1 as components
from utils import (
    get_em3_fig,
    get_em6_fig,
    get_em7_fig,
    get_gender_fig,
    save_audio,
    get_melspec,
    spectrogram,
)
from streamlit_theme import st_theme

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
    st.sidebar.subheader("Display options")

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
    elif st.button("Try test file"):
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
                st.write(
                    spectrogram(
                        "MFCCs",
                        theme["backgroundColor"],
                        theme["textColor"],
                        mfccs,
                        sr,
                        "time",
                        "mel",
                    )
                )
            with col2:
                st.write(
                    spectrogram(
                        "Mel-log-spectrogram",
                        theme["backgroundColor"],
                        theme["textColor"],
                        Xdb,
                        sr,
                        "time",
                        "hz",
                    )
                )

        st.markdown("## Predictions")
        grid = [[], []]
        with st.container():
            grid[0] = st.columns(2)
        with st.container():
            grid[1] = st.columns(2)
        selected = []
        if em3:
            selected.append("em3")
        if em6:
            selected.append("em6")
        if em7:
            selected.append("em7")
        if gender:
            selected.append("gender")

        for i in range(2):
            for j in range(2):
                if len(selected) > 0:
                    if selected[0] == "em3":
                        grid[i][j].write(
                            get_em3_fig(
                                path, theme["backgroundColor"], theme["textColor"]
                            )
                        )
                        selected.pop(0)
                    elif selected[0] == "em6":
                        grid[i][j].write(
                            get_em6_fig(
                                path, theme["backgroundColor"], theme["textColor"]
                            )
                        )
                        selected.pop(0)
                    elif selected[0] == "em7":
                        grid[i][j].write(
                            get_em7_fig(
                                path, theme["backgroundColor"], theme["textColor"]
                            )
                        )
                        selected.pop(0)
                    elif selected[0] == "gender":
                        grid[i][j].write(
                            get_gender_fig(
                                path, theme["backgroundColor"], theme["textColor"]
                            )
                        )
                        selected.pop(0)


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

    df = pd.read_csv("audio_data_info.csv")
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
