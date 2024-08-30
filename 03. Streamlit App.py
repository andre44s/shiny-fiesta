#import library
import streamlit as st
import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from openai import OpenAI
st.set_page_config(page_title="Spotify Review Bot", page_icon="🚀", layout="wide")

# function to dynamically create index for ANN search
# able to filter feature by date, ratings, and app version
@st.cache_data
def create_faiss_index(start_date, end_date, selected_ratings, selected_versions, df, embeddings):

    # convert date
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    df['review_timestamp'] = pd.to_datetime(df['review_timestamp'])

    if len(selected_ratings) == 0:
        selected_ratings = [1, 2, 3, 4, 5]

    if len(selected_versions) == 0:
        selected_versions = df['author_app_version'].unique()

    # filtering dataframe
    filtered_df = df[
        (df['review_timestamp'] >= start_date) &
        (df['review_timestamp'] <= end_date) &
        (df['review_rating'].isin(selected_ratings)) &
        (df['author_app_version'].isin(selected_versions))
    ]

    # filter embeddings based on dataframe index
    filtered_embeddings = embeddings[filtered_df.index]

    # build FAISS ANN index
    d = filtered_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(filtered_embeddings)

    return index, filtered_df

# function to do ANN search using FAISS
def get_relevant_reviews(question, index, model, df, k=200):
    # encode the question to embeddings
    question_embedding = model.encode(question)
    
    #perform ANN search using FAISS
    distances, indices = index.search(np.array([question_embedding]), k)
    
    # filter dataframe with the correct index
    relevant_reviews = df.iloc[indices[0]]
    
    return relevant_reviews

def query_chatgpt(question, relevant_reviews, chat_history):
    # combine the question with reviews
    context = ""
    for index, row in relevant_reviews.iterrows():
        context += f"Review Date: {row['review_timestamp']}, Rating: {row['review_rating']} stars, App Version : {row['author_app_version']}\n"
        context += f"Review: {row['cleaned_review_text']}\n\n"
    
    # system message for the LLM
    messages = [{
        "role": "system",
        "content": '''
            You are a highly advanced and intelligent system specifically designed to analyze and provide insights based on a 
            large dataset of user reviews for Spotify. Your primary goal is to help users understand what these reviews reveal 
            about user experiences, preferences, concerns, and satisfaction levels with Spotify.

            When responding to questions, answer strictly and directly based on the provided review data. Do not include any 
            additional insights, suggestions, or improvements unless explicitly requested by the user.

            You have access to the review data, including the text, star ratings, and timestamps, which should be used to 
            support your analysis. Do not use any external knowledge or information not contained within the provided data. 
            Your responses should be accurate, concise, and focused solely on the user's query, 
            based entirely on the data available through the RAG system.
        '''
    }]

    # if chat history is not empty add it on the promt as well
    if chat_history:
        for message in chat_history:
            role = "user" if message["role"] == "user" else "assistant"
            messages.append({"role": role, "content": message["content"]})
    
    # user's question
    messages.append({"role": "user", "content": f"{question}\n\nRelevant Information:\n{context}"})

    # call openai
    response = st.session_state.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content

st.title("Spotify Review Bot")

# initial loading of necessary informatin like embeddings, df, model, etc
# only done once
if "model" not in st.session_state:
    with st.spinner('Loading resources...'):
        # load embeddigns model
        st.session_state.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        # load embeddings and review data
        st.session_state.embeddings = np.load('data/embeddings.npy')
        st.session_state.df = pd.read_csv('data/Cleaned Text.csv')
        st.session_state.df['review_timestamp'] = pd.to_datetime(st.session_state.df['review_timestamp'])

        # calculate min and max date for filtering
        st.session_state.min_date = st.session_state.df['review_timestamp'].min().date()
        st.session_state.max_date = st.session_state.df['review_timestamp'].max().date()

        # initialize openai client
        st.session_state.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# sidebar will serve as a feature to filter the reviews
with st.sidebar:
    # filter date
    selected_date = st.date_input(
        "Select Review Dates",
        (st.session_state.min_date, st.session_state.max_date),
        st.session_state.min_date,
        st.session_state.max_date,
        format="DD.MM.YYYY",
    )

    # filter app version
    unique_versions = sorted(st.session_state.df['author_app_version'].dropna().unique(), reverse=True)
    selected_versions = st.multiselect("Select App Versions", unique_versions)

    # filter ratings
    st.write("Review Stars: ")
    selected_ratings = []
    if st.checkbox("1 Star", value=True):
        selected_ratings.append(1)
    if st.checkbox("2 Stars", value=True):
        selected_ratings.append(2)
    if st.checkbox("3 Stars", value=True):
        selected_ratings.append(3)
    if st.checkbox("4 Stars", value=True):
        selected_ratings.append(4)
    if st.checkbox("5 Stars", value=True):
        selected_ratings.append(5)


# initialize chat history
# only done once
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history first if any
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# user's question
if prompt := st.chat_input("Ask a question about Spotify reviews:"):
    # display user question
    with st.chat_message("user"):
        st.markdown(prompt)
    # add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # loading indicator based on the question trying to do ANN search, 
    # and then feed the information into LLM 
    with st.spinner('Finding relevant reviews and generating response...'):
        filtered_ann_index, filtered_df = create_faiss_index(selected_date[0], selected_date[1], selected_ratings, selected_versions, st.session_state.df, st.session_state.embeddings)
        relevant_reviews = get_relevant_reviews(prompt, filtered_ann_index, st.session_state.model, filtered_df)
        answer = query_chatgpt(prompt, relevant_reviews, st.session_state.messages)

    # display the LLM result
    with st.chat_message("assistant"):
        st.markdown(answer)
    # add LLM message to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})

