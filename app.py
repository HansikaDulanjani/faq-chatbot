import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

st.set_page_config(page_title="Grocery FAQ Bot", page_icon="🛒")
st.title("🛒 FreshMart FAQ Assistant")
st.caption("Ask me anything about ordering, delivery, payments, or returns!")

@st.cache_resource
def load_everything():
    df = pd.read_csv("faq_data.csv")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['question'].tolist()).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return df, model, index

df, model, index = load_everything()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def search_faq(user_question, top_k=1):
    query_embedding = model.encode([user_question]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    idx = indices[0][0]
    return df.iloc[idx]['question'], df.iloc[idx]['answer']

def generate_reply(matched_answer, user_question):
    prompt = f"""A customer asked: "{user_question}"
The stored answer is: "{matched_answer}"

Rephrase this answer in a warm, natural, and polite tone for a customer support chat. Keep it short and clear."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    matched_q, matched_a = search_faq(user_input)
    reply = generate_reply(matched_a, user_input)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)
