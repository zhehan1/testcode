import os, streamlit as st
st.write("🚀 Root files:", os.listdir("."))
import sys, subprocess
st.write("📦 Installed packages:")
freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode().splitlines()
cv_pkgs = [p for p in freeze if "opencv" in p.lower()]
st.write(cv_pkgs)
import cv2
st.write("✅ cv2 loaded, version:", cv2.__version__)
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import random
import os
import plotly.express as px
from emotion_utils.detector import EmotionDetector
import tempfile
from location_utils.extract_gps import extract_gps, convert_gps
from location_utils.geocoder import get_address_from_coords
from location_utils.landmark import load_models,detect_landmark, query_landmark_coords,LANDMARK_KEYWORDS


# ----------------- App Configuration -----------------
st.set_page_config(
    page_title="AI Emotion & Location Detector",
    page_icon="👁‍🗨",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_detector():
    return EmotionDetector()

detector = get_detector()

def save_history(username, emotion, confidence, location):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[username, emotion, confidence, location, now]], 
                     columns=["Username","Emotion","Confidence","Location","timestamp"])
    try:
        if os.path.exists("history.csv"):
            prev = pd.read_csv("history.csv")
            df = pd.concat([prev, df])
        df.to_csv("history.csv", index=False)
    except Exception as e:
        st.error(f"Failed to save history: {e}")

def show_detection_guide():
    with st.expander("ℹ️ How Emotion Detection Works", expanded=False):
        st.markdown("""
        *Detection Logic Explained:*
        - 😊 Happy: Smile present, cheeks raised
        - 😠 Angry: Eyebrows lowered, eyes wide open
        - 😐 Neutral: No strong facial movements
        - 😢 Sad: Eyebrows raised, lip corners down
        - 😲 Surprise: Eyebrows raised, mouth open
        - 😨 Fear: Eyes tense, lips stretched
        - 🤢 Disgust: Nose wrinkled, upper lip raised

        *Tips for Better Results:*
        - Use clear, front-facing images
        - Ensure good lighting
        - Avoid obstructed faces
        """)

def sidebar_design(username):
    if username:
        st.sidebar.success(f"👤 Logged in as: {username}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Quick Navigation")
    st.sidebar.markdown("- Upload and detect emotions")
    st.sidebar.markdown("- View and filter upload history")
    st.sidebar.markdown("- Visualize your emotion distribution")
    st.sidebar.divider()
    st.sidebar.info("Enhance your experience by ensuring clear, well-lit facial images.")
  
def main():
    st.title("👁‍🗨 AI Emotion & Location Detector")
    st.caption("Upload a photo to detect facial emotions and estimate location.")
    tabs = st.tabs(["🏠 Home", "🗺️ Location Map", "📜 Upload History", "📊 Emotion Analysis Chart"])

    with tabs[0]:
        username = st.text_input("👤 Enter your username")
        sidebar_design(username)
        if username:
            uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png"])
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name
               
                try:
                    # 调试信息
                    print(f"[MAIN] Processing image: {uploaded_file.name}")
                    print(f"[MAIN] Image size: {uploaded_file.size} bytes")
                    
                    image = Image.open(temp_path).convert("RGB")
                    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    detections = detector.detect_emotions(img)
                    detected_img = detector.draw_detections(img, detections)
                        
                    location = "Unknown"
                    method = ""
                    gps_info = extract_gps(temp_path)
                
                    if gps_info:
                        print(f"[MAIN] GPS extraction successful: {list(gps_info.keys())}")
                        coords = convert_gps(gps_info)
   
                        if coords:
                            print(f"[MAIN] GPS coordinates: {coords}")
                            location = get_address_from_coords(coords)
                            method="GPS Metadata"
                        else:
                            print("[MAIN] No GPS data found in image")
                                
                    if location in ("Unknown", "Unknown location"):
                        print("[MAIN] Trying landmark detection...")
                        landmark = detect_landmark(temp_path, threshold=0.15, top_k=5)
                        
                        if landmark:
                            print(f"[MAIN] CLIP predicted landmark: {landmark}")
                            st.write(f"🔍 CLIP predicted landmark: **{landmark}**")
                            
                            coords_result, source = query_landmark_coords(landmark)
                            
                            if coords_result:
                                lat, lon = coords_result
                                print(f"[MAIN] Landmark coordinates: {lat}, {lon} (source: {source})")
                                addr = get_address_from_coords((lat, lon))
                                if addr and addr not in ("Unknown location", "Invalid coordinates", "Geocoding service unavailable"):
                                    location = addr
                                    method = f"Landmark ({source})"
                                    print(f"[MAIN] Final location: {location}")
                                else:
                                    if landmark in LANDMARK_KEYWORDS:
                                        landmark_info = LANDMARK_KEYWORDS[landmark]
                                        location = f"{landmark_info[0]}, {landmark_info[1]}"
                                    else:
                                        location = f"{landmark.title()} ({lat:.4f}, {lon:.4f})"
                                        method = f"Landmark ({source})"
                                        print(f"[MAIN] Using landmark fallback: {location}")
                            else:
                                print(f"[MAIN] No coordinates found for landmark: {landmark}")
                                st.write(f"⚠️ Landmark detected but no coordinates available")
                        else:
                            print("[MAIN] No landmark detected with sufficient confidence")
                            st.write("🔍 No landmark detected with sufficient confidence")
                except Exception as e:
                    st.error(f"❌ Something went wrong during processing: {e}")
                    print(f"[ERROR] {e}")
                                   
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.subheader("🔍 Detection Results")
                    if detections:
                        emotions = [d["emotion"] for d in detections]
                        confidences = [d["confidence"] for d in detections]
                        st.success(f"🎭 {len(detections)} face(s) detected")
                        for i, (emo, conf) in enumerate(zip(emotions, confidences)):
                            st.write(f"- Face {i + 1}: {emo} ({conf}%)")
                        show_detection_guide()
                        st.write(f"📍 Estimated Location: **{location}** ({method})")
                        save_history(username, emotions[0], confidences[0], location)
                    else:
                        st.warning("No faces were detected in the uploaded image.")
                with col2:
                    t1, t2 = st.tabs(["Original Image", "Processed Image"])
                    with t1:
                         st.image(image, use_container_width=True)
                    with t2:
                        st.image(detected_img, channels="BGR", use_container_width=True,
                                    caption=f"Detected {len(detections)} face(s)")
               

    with tabs[1]:
        st.subheader("🗺️ Random Location Sample (Demo)")
        st.map(pd.DataFrame({
            'lat': [3.139 + random.uniform(-0.01, 0.01)],
            'lon': [101.6869 + random.uniform(-0.01, 0.01)]
        }))
        st.caption("Note: This location map is a demo preview and not actual detected GPS data.")

    with tabs[2]:
        st.subheader("📜 Upload History")
        if username:
            try:
                if os.path.exists("history.csv"):
                    df = pd.read_csv("history.csv")
                    if df.empty:
                        st.info("No upload records found.")
                    else:
                        df_filtered = df[df["Username"].str.contains(username, case=False)]
                        df_filtered = df_filtered.sort_values("timestamp", ascending=False).reset_index(drop=True)
                        df_filtered.index = range(1, len(df_filtered)+1)
                        st.dataframe(df_filtered)
                        st.caption(f"Total records found for {username}: {len(df_filtered)}")
                else:
                    st.info("No history file found.")
            except:
                st.warning("Error loading history records.")
        else:
            st.warning("Please enter your username to view your upload history.")

    with tabs[3]:
        st.subheader("📊 Emotion Analysis Chart")
        if username:
            try:
                if os.path.exists("history.csv"):
                    df = pd.read_csv("history.csv")
                    df_filtered = df[df["Username"].str.contains(username, case=False)]
                    if not df_filtered.empty:
                        fig = px.pie(df_filtered, names="Emotion", title=f"Emotion Distribution for {username}")
                        st.plotly_chart(fig)
                        st.caption("Chart is based on your personal upload history.")
                    else:
                        st.info("No emotion records found for this username.")
                else:
                    st.info("History file not found.")
            except Exception as e:
                st.error(f"Error generating chart: {e}")
        else:
            st.warning("Please enter your username to generate your emotion chart.")

if __name__ == "__main__":
    main()
