
import re
import string
 
import joblib
import streamlit as st
 
# ---------------------------------------------------------------
# Page config + dark theme styling
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Speech-to-Text NLP Prediction",
    page_icon="🎤",
    layout="centered",
)
 
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #0f1117 0%, #161a25 100%);
        color: #e8e8e8;
    }
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #7f5af0, #2cb67d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        text-align: center;
        color: #9a9ca5;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background: #1c2030;
        border: 1px solid #2a2f45;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
    }
    .pred-positive { color: #2cb67d; font-weight: 700; font-size: 1.4rem; }
    .pred-negative { color: #ef4565; font-weight: 700; font-size: 1.4rem; }
    .pred-neutral  { color: #f2c14e; font-weight: 700; font-size: 1.4rem; }
    .stButton>button {
        background: linear-gradient(90deg, #7f5af0, #2cb67d);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
st.markdown('<div class="main-title">🎤 Speech-to-Text NLP Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Speak or type a review — get the sentiment prediction and confidence score.</div>', unsafe_allow_html=True)
 
 
# ---------------------------------------------------------------
# Step 16: Load the Saved Model
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer
 
try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python train_pipeline.py` (or the notebook) "
        "first to generate best_model.pkl and vectorizer.pkl."
    )
    st.stop()
 
 
# ---------------------------------------------------------------
# Preprocessing (must match model_building.ipynb exactly)
# ---------------------------------------------------------------
import nltk
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)
 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
 
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
 
 
def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)
 
 
def predict_sentiment(raw_text: str):
    processed = preprocess_text(raw_text)
    vec = vectorizer.transform([processed])
    prediction = model.predict(vec)[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)
        confidence = proba.max() * 100
    return prediction, confidence, processed
 
 
def render_result(recognized_text, prediction, confidence, processed_text):
    css_class = {
        "Positive": "pred-positive",
        "Negative": "pred-negative",
        "Neutral": "pred-neutral",
    }.get(prediction, "pred-neutral")
 
    conf_html = f"<p>Confidence: <b>{confidence:.2f}%</b></p>" if confidence is not None else ""
 
    st.markdown(
        f"""
        <div class="result-card">
            <p>Recognized Text:</p>
            <p style="color:#c9cbd6;">"{recognized_text}"</p>
            <p>Prediction: <span class="{css_class}">{prediction}</span></p>
            {conf_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Preprocessed text sent to the model"):
        st.code(processed_text or "(empty after preprocessing)")
 
 
# ---------------------------------------------------------------
# Tabs: Speech-to-Text prediction | Optional MediaPipe gesture detection
# ---------------------------------------------------------------
tab_speech, tab_gesture = st.tabs(["🎤 Speech-to-Text Prediction", "✋ Gesture Detection (optional)"])
 
with tab_speech:
    st.write("Speak into the microphone, or type text below, to get a sentiment prediction.")
 
    input_mode = st.radio("Input method", ["🎙️ Record from microphone", "⌨️ Type text"], horizontal=True)
 
    # -----------------------------------------------------------
    # Step 18: Speech-to-Text
    # -----------------------------------------------------------
    if input_mode == "🎙️ Record from microphone":
        st.caption(
            "Records audio from **your browser's microphone** (works on the deployed app too — "
            "your browser will ask permission the first time)."
        )
        try:
            from streamlit_mic_recorder import mic_recorder
 
            # format="wav" makes the component decode the browser's raw recording
            # (WebM/Opus) into PCM WAV on the JS side and hand back raw sample
            # bytes + sample_rate + sample_width, which we feed straight into
            # speech_recognition's AudioData - this avoids ever needing to parse
            # a WebM/Opus file on the Python side.
            audio_dict = mic_recorder(
                start_prompt="🎤 Start recording",
                stop_prompt="⏹️ Stop recording",
                just_once=True,
                use_container_width=True,
                format="wav",
                key="mic_recorder",
            )
 
            if audio_dict is not None and audio_dict.get("bytes"):
                try:
                    import speech_recognition as sr
 
                    recognizer = sr.Recognizer()
                    audio_data = sr.AudioData(
                        audio_dict["bytes"],
                        audio_dict["sample_rate"],
                        audio_dict["sample_width"],
                    )
                    # audio_dict["bytes"] is raw PCM (no WAV header) - build a
                    # proper playable WAV file for the <audio> player.
                    st.audio(audio_data.get_wav_data(), format="audio/wav")
 
                    with st.spinner("Transcribing..."):
                        recognized_text = recognizer.recognize_google(audio_data)
 
                    prediction, confidence, processed = predict_sentiment(recognized_text)
                    render_result(recognized_text, prediction, confidence, processed)
 
                except ImportError:
                    st.error("Install the `SpeechRecognition` package to use this feature.")
                except sr.UnknownValueError:
                    st.warning("Couldn't understand the audio — try speaking a bit louder/clearer and re-record.")
                except Exception as e:
                    st.error(f"Could not recognize speech from the recording: {e}")
 
        except ImportError:
            st.error(
                "The `streamlit-mic-recorder` package isn't installed — "
                "add it to requirements.txt to enable browser mic recording. "
                "Meanwhile, please upload a WAV file or type your text instead."
            )
 
        st.markdown("---")
        st.caption("Or upload a short audio clip (WAV) instead of using the mic:")
        audio_file = st.file_uploader("Upload WAV audio", type=["wav"])
        if audio_file is not None and st.button("Transcribe uploaded audio"):
            try:
                import speech_recognition as sr
 
                recognizer = sr.Recognizer()
                with sr.AudioFile(audio_file) as source:
                    audio = recognizer.record(source)
                recognized_text = recognizer.recognize_google(audio)
                prediction, confidence, processed = predict_sentiment(recognized_text)
                render_result(recognized_text, prediction, confidence, processed)
            except ImportError:
                st.error("Install the `SpeechRecognition` package to use this feature.")
            except Exception as e:
                st.error(f"Could not transcribe the uploaded file: {e}")
 
    else:
        typed_text = st.text_area("Enter text (simulates recognized speech):", placeholder="e.g. I am really happy with the product.")
        if st.button("Predict"):
            if typed_text.strip():
                prediction, confidence, processed = predict_sentiment(typed_text)
                render_result(typed_text, prediction, confidence, processed)
            else:
                st.warning("Please enter some text first.")
 
with tab_gesture:
    st.write("Optional bonus feature: real-time hand-gesture detection using MediaPipe + OpenCV.")
    st.caption(
        "This uses your webcam and only runs locally (Streamlit + OpenCV cannot access a "
        "camera in a hosted/cloud environment without extra plumbing)."
    )
 
    run_gesture = st.checkbox("Start webcam gesture detection")
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
 
    HAND_MODEL_PATH = "hand_landmarker.task"
    HAND_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )
 
    if run_gesture:
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
 
            # MediaPipe's legacy `mp.solutions` namespace was removed in
            # mediapipe 0.10.31+, so this uses the current Tasks API
            # (HandLandmarker) instead - same approach as the other MediaPipe
            # projects in this workspace.
            if not __import__("os").path.exists(HAND_MODEL_PATH):
                with st.spinner("Downloading hand-landmarker model (first run only)..."):
                    import urllib.request
                    urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
 
            base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_tracking_confidence=0.6,
                running_mode=mp_vision.RunningMode.VIDEO,
            )
            landmarker = mp_vision.HandLandmarker.create_from_options(options)
 
            HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS if hasattr(mp, "solutions") else None
 
            cap = cv2.VideoCapture(0)
            stop_button = st.button("Stop")
            frame_ts = 0
            while cap.isOpened() and run_gesture and not stop_button:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
 
                frame_ts += 1
                result = landmarker.detect_for_video(mp_image, frame_ts)
 
                status = "No hand detected"
                h, w, _ = frame.shape
                if result.hand_landmarks:
                    status = f"Hand detected ({len(result.hand_landmarks)})"
                    for hand in result.hand_landmarks:
                        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                        for x, y in pts:
                            cv2.circle(frame, (x, y), 3, (44, 182, 125), -1)
                        if HAND_CONNECTIONS:
                            for a, b in HAND_CONNECTIONS:
                                cv2.line(frame, pts[a], pts[b], (127, 90, 240), 2)
 
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                status_placeholder.write(status)
 
            cap.release()
            landmarker.close()
        except ImportError:
            st.error("Install `opencv-python` and `mediapipe` to use gesture detection.")
        except Exception as e:
            st.error(f"Webcam/gesture detection error: {e}")
 
st.markdown("---")
st.caption("Speech-to-Text Prediction Using NLP · Innomatics Research Labs Final Project")
