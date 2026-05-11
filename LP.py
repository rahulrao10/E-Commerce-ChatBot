import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



data = pd.read_csv("Ecommerce_FAQs.csv")
data.columns = data.columns.str.strip().str.lower()

questions = data["prompt"].tolist()
answers = data["response"].tolist()


vectorizer = TfidfVectorizer(stop_words="english")
question_vectors = vectorizer.fit_transform(questions)



def chatbot_response(user_input):

    user_input = user_input.lower()

    if user_input in ["hi", "hello", "hey"]:
        return "Hello! How can I help you today?"

    if user_input in ["bye", "exit", "quit"]:
        return "Thank you for chatting! Have a great day 😊"

    user_vector = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vector, question_vectors)

    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[0][best_match_index]

    if best_score < 0.2:
        return "Sorry, I could not understand your question. Please try again."

    return answers[best_match_index]


st.markdown("""
    <style>
        /* Page background */
        body {
            background-color: #f2f2f2;
        }

        /* Text input box */
        .stTextInput > div > div > input {
            background-color: #e6e6e6;
            color: black;              /* typed text */
            caret-color: black;        /* cursor color */
        }

        /* User message box */
        .chat-box {
            background-color: #d9d9d9;
            color: black;              /* user text */
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
        }

        /* Bot message box */
        .bot {
            background-color: #bfbfbf;
            color: black;              /* bot answer text */
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)



st.title("🛒 E-Commerce FAQ Chatbot")
st.write("Ask any question related to our e-commerce services.")
if st.button("🧹 Clear Conversation"):
    st.session_state.chat_history = []



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


user_input = st.text_input("You:", "")

if user_input:
    bot_reply = chatbot_response(user_input)

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", bot_reply))


for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(
            f"<div class='chat-box'><b>You:</b> {message}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='bot'><b>Bot:</b> {message}</div>",
            unsafe_allow_html=True
        )
