import streamlit as st
import nltk
import random
import json
import pickle
import string
import numpy as np
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Load resources
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Helper functions
def clean_up_sentence(sentence):
    #sentence = sentence.lower()#NEW
    #sentence = sentence.translate(str.maketrans('', '', string.punctuation))#NEW
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(return_list, intents_json):
    if len(return_list) == 0:
        tag = 'noanswer'
    else:
        tag = return_list[0]['intent']

    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if tag == i['tag']:
            result = random.choice(i['responses'])
    return result

def chatbot_response(text):
    return_list = predict_class(text)
    response = get_response(return_list, intents)
    return response


# Initialize chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("MindMate Chatbot")

# Display chat messages
messages = st.container(height=400)

# Show chat history in the container
with messages:
    for message in st.session_state.chat_history:
        st.chat_message(message['role']).write(message['content'])

#icon=":material/thumb_up:"

# User input
if prompt := st.chat_input("Say something"):
    # Save user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Get the bot response
    bot_response = chatbot_response(prompt)

    # Save bot response to chat history
    st.session_state.chat_history.append({"role": "ai", "content": bot_response})

    # Display the user's message
    messages.chat_message("user").write(prompt)

    # Display the bot's response
    messages.chat_message("ai").write(f"MindMate: {bot_response}")
