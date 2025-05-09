import os
import sys
import pandas as pd
import altair as alt

import io
import streamlit as st
from fake_audio_detection.model import predict_audio_blocks

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


st.title("🔎 DeepVoice Detection")

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# if you want to code your training part
DATASET_DIR = os.path.join(APP_DIR, "dataset/")
MODEL_PATH = os.path.join(APP_DIR, "model/noma-1")

REAL_DIR = os.path.join(APP_DIR, "audios/real")
FAKE_DIR = os.path.join(APP_DIR, "audios/fake")

# Then continue as before
real_audio = {
    f"Real - {f}": os.path.join(REAL_DIR, f)
    for f in os.listdir(REAL_DIR)
    if f.endswith((".wav", ".mp3"))
}
fake_audio = {
    f"Fake - {f}": os.path.join(FAKE_DIR, f)
    for f in os.listdir(FAKE_DIR)
    if f.endswith((".wav", ".mp3"))
}
all_audio = {**real_audio, **fake_audio}

selected_label = st.radio("Select an audio file to play:", list(all_audio.keys()))
selected_path = all_audio[selected_label]

st.write("#### Try with your audios")
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])

selected_label = "Default Audio"

if uploaded_file is not None:
    st.markdown(f"**Now Playing:** `{uploaded_file.name}`")
    audio_bytes = uploaded_file.read()
    file_extension = uploaded_file.name.split(".")[-1].lower()
    st.audio(audio_bytes, format=f"audio/{file_extension}")
else:
    st.markdown(f"**Now Playing:** `{selected_label}`")
    with open(selected_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")


if st.button("Run Prediction") and os.path.exists(MODEL_PATH):
    audio_bytes = None
    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        audio_bytes = io.BytesIO(bytes_data)
    with st.spinner("Analyzing audio..."):
        times, probas = predict_audio_blocks(MODEL_PATH, selected_path, audio_bytes)

        preds = probas.argmax(axis=1)
        confidences = probas.max(axis=1)
        preds_as_string = ["Fake" if i == 0 else "Real" for i in preds]
        df = pd.DataFrame(
            {"Seconds": times, "Prediction": preds_as_string, "Confidence": confidences}
        )

        def get_color(row):
            if row["Confidence"] < 0.3:
                return "Uncertain"
            return row["Prediction"]

        df["Confidence Level"] = df.apply(get_color, axis=1)

        # Plot
        st.markdown("### Prediction by 1s Blocks")
        st.markdown(
            "Hover above each bar to see the confidence level of each prediction."
        )
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Seconds:O", title="Seconds"),
                y=alt.value(30),
                color=alt.Color(
                    "Confidence Level:N",
                    scale=alt.Scale(
                        domain=["Fake", "Real", "Uncertain"],
                        range=["steelblue", "green", "gray"],
                    ),
                ),
                tooltip=["Seconds", "Prediction", "Confidence"],
            )
            .properties(width=700, height=150)
        )

        text = (
            alt.Chart(df)
            .mark_text(
                align="right",
                baseline="top",
                dy=10,
                color="white",
                xOffset=10,
                yOffset=-20,
                fontSize=14,
            )
            .encode(x=alt.X("Seconds:O"), y=alt.value(15), text="Prediction:N")
        )

        st.altair_chart(chart + text, use_container_width=True)

        st.markdown("### Overall prediction")
        if all(element == "Real" for element in preds_as_string):
            st.markdown("The audio is **Real**")
        elif all(element == "Fake" for element in preds_as_string):
            st.markdown("The audio is **Fake**")
        else:
            st.markdown("Some parts of the audio have been detected as **Fake**")

elif not os.path.exists(MODEL_PATH):
    st.warning(f"Missing model: {MODEL_PATH}")
