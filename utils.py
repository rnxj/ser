import os
import numpy as np
import cv2
import librosa
import librosa.display
from keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image

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


def get_melspec(audio):
    y, sr = librosa.load(audio, sr=44100)
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    img = np.stack((Xdb,) * 3, -1)
    img = img.astype(np.uint8)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (224, 224))
    rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
    return (rgbImage, Xdb)


def get_title(predictions, categories, first_line=""):
    txt = f"{first_line}\nDetected emotion: \
  {categories[predictions.argmax()]} - {predictions.max() * 100:.2f}%"
    return txt


def plot_colored_polar(
    fig,
    predictions,
    categories,
    title="",
    colors=COLOR_DICT,
    facecolor="lightgrey",
    fontcolor="darkblue",
):
    N = len(predictions)
    ind = predictions.argmax()

    COLOR = color_sector = colors[categories[ind]]
    sector_colors = [colors[i] for i in categories]

    fig.set_facecolor(facecolor)
    plt.rcParams["text.color"] = fontcolor
    ax = plt.subplot(111, polar="True")

    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    for sector in range(predictions.shape[0]):
        radii = np.zeros_like(predictions)
        radii[sector] = predictions[sector] * 10
        width = np.pi / 1.8 * predictions
        c = sector_colors[sector]
        ax.bar(theta, radii, width=width, bottom=0.0, color=c, alpha=0.25)

    angles = [i / float(N) * 2 * np.pi for i in range(N)]
    angles += angles[:1]

    data = list(predictions)
    data += data[:1]
    plt.polar(angles, data, color=COLOR, linewidth=2)
    plt.fill(angles, data, facecolor=COLOR, alpha=0.25)

    ax.spines["polar"].set_color(facecolor)
    ax.set_theta_offset(np.pi / 3)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color=fontcolor, size=8)
    ax.set_rlabel_position(0)
    plt.yticks([0, 0.25, 0.5, 0.75, 1], color="grey", size=8)

    plt.suptitle(title, color=fontcolor, fontsize=12)
    plt.title(f"BIG {N}\n", color=fontcolor, fontsize=10)
    plt.ylim(0, 1)
    plt.subplots_adjust(top=0.75)


def plot_melspec(path, tmodel=None, three=False, CAT3=CAT3, CAT6=CAT6):
    # load model if it is not loaded
    if tmodel is None:
        tmodel = load_model("models/melspec.h5")
    # mel-spec model results
    mel = get_melspec(path)[0]
    mel = mel.reshape(1, *mel.shape)
    tpred = tmodel.predict(mel)[0]
    cat = CAT6

    if three:
        pos = tpred[3] + tpred[5] * 0.5
        neu = tpred[2] + tpred[5] * 0.5 + tpred[4] * 0.5
        neg = tpred[0] + tpred[1] + tpred[4] * 0.5
        tpred = np.array([pos, neu, neg])
        cat = CAT3

    txt = get_title(tpred, cat)
    fig = plt.figure(figsize=(6, 4))
    plot_colored_polar(fig, predictions=tpred, categories=cat, title=txt)
    return (fig, tpred)


def save_audio(file):
    if file.size > 4000000:
        return 1
    folder = "audio"
    datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))

    with open(os.path.join(folder, file.name), "wb") as f:
        f.write(file.getbuffer())
    return 0


def get_melspec(audio):
    y, sr = librosa.load(audio, sr=44100)
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))
    img = np.stack((Xdb,) * 3, -1)
    img = img.astype(np.uint8)
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.resize(grayImage, (224, 224))
    rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
    return (rgbImage, Xdb)


def get_mfccs(audio, limit):
    y, sr = librosa.load(audio)
    a = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if a.shape[1] > limit:
        mfccs = a[:, :limit]
    elif a.shape[1] < limit:
        mfccs = np.zeros((a.shape[0], limit))
        mfccs[:, : a.shape[1]] = a
    return mfccs


def get_title(predictions, categories=CAT6):
    title = f"Detected emotion: {categories[predictions.argmax()]} \
    - {predictions.max() * 100:.2f}%"
    return title


def spectrogram(title, background, textColor, data, sr, x_axis, y_axis):
    fig = plt.figure(figsize=(10, 2))
    fig.set_facecolor(background)
    plt.title(title, color=textColor)
    librosa.display.specshow(data, sr=sr, x_axis=x_axis, y_axis=y_axis)
    plt.gca().get_yaxis().set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    return fig


def get_em3_fig(path, background, textColor):
    model = load_model("models/em3_6.h5")
    mfccs = get_mfccs(path, model.input_shape[-1])
    mfccs = mfccs.reshape(1, *mfccs.shape)
    pred = model.predict(mfccs)[0]
    pos = pred[3] + pred[5] * 0.5
    neu = pred[2] + pred[5] * 0.5 + pred[4] * 0.5
    neg = pred[0] + pred[1] + pred[4] * 0.5
    data3 = np.array([pos, neu, neg])
    txt = "MFCCs\n" + get_title(data3, CAT3)
    em3_fig = plt.figure(figsize=(5, 5))
    plot_colored_polar(
        em3_fig,
        predictions=data3,
        categories=CAT3,
        title=txt,
        colors=COLOR_DICT,
        facecolor=background,
        fontcolor=textColor,
    )
    return em3_fig


def get_em6_fig(path, background, textColor):
    model = load_model("models/em3_6.h5")
    mfccs = get_mfccs(path, model.input_shape[-1])
    mfccs = mfccs.reshape(1, *mfccs.shape)
    pred = model.predict(mfccs)[0]
    txt = "MFCCs\n" + get_title(pred, CAT6)
    em6_fig = plt.figure(figsize=(5, 5))
    plot_colored_polar(
        em6_fig,
        predictions=pred,
        categories=CAT6,
        title=txt,
        colors=COLOR_DICT,
        facecolor=background,
        fontcolor=textColor,
    )
    return em6_fig


def get_em7_fig(path, background, textColor):
    model = load_model("models/em7.h5")
    mfccs = get_mfccs(path, model.input_shape[-2])
    mfccs = mfccs.T.reshape(1, *mfccs.T.shape)
    pred = model.predict(mfccs)[0]
    txt = "MFCCs\n" + get_title(pred, CAT7)
    em7_fig = plt.figure(figsize=(5, 5))
    plot_colored_polar(
        em7_fig,
        predictions=pred,
        categories=CAT7,
        title=txt,
        colors=COLOR_DICT,
        facecolor=background,
        fontcolor=textColor,
    )
    return em7_fig


def get_gender_fig(path, background, textColor):
    gmodel = load_model("models/gender.h5")
    gmfccs = get_mfccs(path, gmodel.input_shape[-1])
    gmfccs = gmfccs.reshape(1, *gmfccs.shape)
    gpred = gmodel.predict(gmfccs)[0]
    gdict = [["female", "woman.png"], ["male", "man.png"]]
    ind = gpred.argmax()
    txt = "Predicted gender: " + gdict[ind][0]
    img = Image.open("images/" + gdict[ind][1])

    gender_fig = plt.figure(figsize=(2, 2))
    gender_fig.set_facecolor(background)
    plt.title(txt, color=textColor, fontsize=8)
    plt.imshow(img)
    plt.axis("off")
    return gender_fig
