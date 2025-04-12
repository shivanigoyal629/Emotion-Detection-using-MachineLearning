import streamlit as st
import joblib

# Load your model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to predict emotion
def predict_emotion(text):
    text_vector = vectorizer.transform([text])
    emotion = model.predict(text_vector)
    return emotion[0]

# Streamlit layout
st.title("Emotion Detection App")
st.write("This app detects emotions from text input.")

user_input = st.text_area("Enter a sentence:")

if user_input:
    predicted_emotion = predict_emotion(user_input)
    st.subheader(f"Predicted Emotion: {predicted_emotion}")

    # Display emoji and GIF based on emotion
    if predicted_emotion == "happy":
        st.write("😊")
        st.image("images/happy_image.gif", caption="You seem happy!", width=150)
    elif predicted_emotion == "sad":
        st.write("😢")
        st.image("images/sad_image.gif", caption="Feeling sad?", width=150)
    elif predicted_emotion == "angry":
        st.write("😡")
        st.image("images/angry_image.gif", caption="Angry emotion detected", width=150)
    else:
        st.write("😐")
        st.image("images/neutral_image.gif", caption="Neutral emotion detected", width=150)

    # You can remove or comment out the sound-playing code if you don't want sound
    # play_emotion_sound(predicted_emotion)
