# AI-Powered Business FAQ Chatbot

An AI chatbot that answers customer FAQs for a Sri Lankan online grocery store, built to demonstrate applied NLP and semantic search — not just keyword matching.

## What it does
Customers can ask questions in their own words (e.g. "when will my groceries arrive?") and the bot understands the *meaning* behind the question, matches it to the closest known answer, and replies in a natural, conversational tone.

## How it works
1. **Data**: 100 realistic Q&A pairs covering ordering, delivery, payments, returns, account, products, and promotions.
2. **Embeddings**: Each question is converted into a 384-dimension vector using `sentence-transformers` (`all-MiniLM-L6-v2`), capturing its meaning rather than exact words.
3. **Semantic Search**: FAISS indexes these embeddings and instantly finds the closest matching question to a new user query, even with completely different wording.
4. **Natural Language Generation**: The matched answer is passed to Groq's LLaMA 3.1 model, which rephrases it into a warm, natural customer-support style reply.
5. **Interface**: Built with Streamlit for a simple, real-time chat experience.

## Tech Stack
Python · Sentence Transformers · FAISS · Groq API · Streamlit · Pandas

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
You'll need a free Groq API key set as an environment variable or Streamlit secret: `GROQ_API_KEY`.

## Live Demo
[[Streamlit Cloud link](https://faq-chatbot-jd5enis5lk8w43s6ssxj3p.streamlit.app/)]

## Future Improvements
- Expand the FAQ dataset for broader coverage
- Add multi-language support (Sinhala/Tamil)
- Log unanswered questions to identify gaps in the knowledge base
