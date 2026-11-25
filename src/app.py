import sys
from pathlib import Path

import streamlit as st
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from hybrid_rag import ask_hybrid_rag  # noqa: E402

st.set_page_config(page_title="TN Govt Schemes Chatbot")
st.title("Tamil Nadu Government Schemes Chatbot")

client = OpenAI()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

audio_val = st.audio_input("Record your question")
text_val = st.chat_input("Or type your question...")

prompt = None
if audio_val is not None:
    audio_bytes = audio_val.getvalue()
    audio_id = f"{len(audio_bytes)}_{hash(audio_bytes[:64])}"
    if st.session_state.get("last_audio_id") != audio_id:
        st.session_state.last_audio_id = audio_id
        audio_val.seek(0)
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_val,
        )
        prompt = transcript.text

if text_val:
    prompt = text_val

if prompt and prompt.strip():
    clean_prompt = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": clean_prompt})
    with st.chat_message("user"):
        st.markdown(clean_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            result = ask_hybrid_rag(clean_prompt, chat_history=st.session_state.messages[:-1])
            answer = result["answer"] if isinstance(result, dict) else str(result)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})