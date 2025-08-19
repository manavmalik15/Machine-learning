# app.py
import streamlit as st

# rag_pipelin

from huggingface_hub import InferenceClient


import os
#!pip install PyMuPDF
import fitz
#!pip install faiss-cpu
import faiss
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from sklearn.metrics.pairwise import cosine_similarity


chicago_tz = ZoneInfo("America/Chicago")

class RAGChatbot:
    def __init__(
        self,
        pdf_folder: str = '/Users/manavmalik/Desktop/bot_resorces/doc',
        chunk_size: int = 500,
        overlap: int = 0,
        top_k: int = 3,
        hf_api_key: str = "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",
        hf_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    ):
        # Configuration
        self.pdf_folder  = pdf_folder
        self.chunk_size  = chunk_size
        self.overlap     = overlap
        self.top_k       = top_k
        self.hf_api_key  = hf_api_key
        self.hf_model    = hf_model

        # 1) Load & chunk
        texts = self._extract_texts()
        self.chunks = [c for t in texts for c in self._chunk(t)]

        # 2) Embed
        self.embedder   = SentenceTransformer("all-mpnet-base-v2")
        embeddings      = self.embedder.encode(self.chunks)

        # 3) Build FAISS index
        dim            = embeddings.shape[1]
        self.index     = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype("float32"))

        # 4) Hugging Face client
        self.client = InferenceClient(
            provider="novita",
            api_key=self.hf_api_key,
        )

    def _extract_texts(self):
        docs = []
        for fn in os.listdir(self.pdf_folder):
            if fn.lower().endswith(".pdf"):
                doc = fitz.open(os.path.join(self.pdf_folder, fn))
                docs.append("".join(p.get_text() for p in doc))
        return docs

    def _chunk(self, text: str):
        words = text.split()
        for i in range(0, len(words), self.chunk_size - self.overlap):
            yield " ".join(words[i : i + self.chunk_size])

    def _retrieve(self, query: str):
        # Step 1: Get top 10 similar chunks
        q_emb, = self.embedder.encode([query])
        _, ids = self.index.search(np.array([q_emb]), 10)
        candidates = [self.chunks[i] for i in ids[0]]

        # Step 2: Rerank these candidates using similarity score
        rerank_scores = []
        for chunk in candidates:
            chunk_emb = self.embedder.encode([chunk])[0]
            score = cosine_similarity([q_emb], [chunk_emb])[0][0]
            rerank_scores.append((score, chunk))

        # Step 3: Sort and return top 3 chunks
        rerank_scores.sort(reverse=True, key=lambda x: x[0])
        top_chunks = [text for _, text in rerank_scores[:self.top_k]]  # self.top_k = 3
        return top_chunks


    def _call_hf(self, context: str, query: str):
        prompt = (
            "Answer the question using the internal document context provided below.\n"
            "=== DOCUMENT CONTEXT START ===\n"
            + context +
            "\n=== DOCUMENT CONTEXT END ===\n"
            f"Question: {query}\n"
            f"(Asked on {datetime.now(chicago_tz).strftime('%Y-%m-%d')})\n"
            "Answer:"
        )
        completion = self.client.chat.completions.create(
            model=self.hf_model,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        # Adapt based on actual return structure
        if hasattr(completion.choices[0].message, "content"):
            return completion.choices[0].message.content
        elif isinstance(completion.choices[0].message, dict):
            return completion.choices[0].message.get("content", "")
        else:
            return str(completion.choices[0].message)


    def _call_hf_structured(self, prompt: str):

        completion = self.client.chat.completions.create(
            model=self.hf_model,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        # Adapt based on actual return structure
        if hasattr(completion.choices[0].message, "content"):
            return completion.choices[0].message.content
        elif isinstance(completion.choices[0].message, dict):
            return completion.choices[0].message.get("content", "")
        else:
            return str(completion.choices[0].message)


    def answer(self, query: str) -> str:
        """Synchronous: retrieve + call Hugging Face + return text."""
        top_chunks = self._retrieve(query)
        context    = "\n\n".join(top_chunks)
        return self._call_hf(context, query)

    def structured_answer(self, query: str , promp: str) -> str:
        top_chunks = self._retrieve(query)
        context    = "\n\n".join(top_chunks)

        prompt = (
            "Answer the question using the internal document context provided below.\n"
            "=== DOCUMENT CONTEXT START ===\n"
            + context +
            "\n=== DOCUMENT CONTEXT END ===\n"

            + promp +

             query +
            f"(Asked on {datetime.now(chicago_tz).strftime('%Y-%m-%d')})\n"
            "Answer:"
        )



        return self._call_hf_structured(prompt)



st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📚 RAG-Powered Chatbot")

# Persist conversation across runs
if "history" not in st.session_state:
    st.session_state.history = []

rag = RAGChatbot()
# Chat input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", "")
    submit = st.form_submit_button("Send")

if submit and user_input:
    bot_answer = rag.answer(user_input)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", bot_answer))

# Render the chat
for speaker, text in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")

# Button to clear history
if st.button("Clear Conversation"):
    st.session_state.history = []

st.markdown("---")
st.markdown("Built with Streamlit and your RAG backend.")
