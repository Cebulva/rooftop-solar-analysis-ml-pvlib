import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("LLM_MODEL")

def initialize_chat():
    if "rag_bot" not in st.session_state:
        st.session_state.rag_bot = {
            "messages": [{"role": "assistant", "content": "How can I assist you today?"}],
            "expanded": False,
            "report_ready": False
        }

def render_chat_interface():
    if not GROQ_KEY:
        st.sidebar.warning("Set GROQ_API_KEY")
        return

    state = st.session_state.rag_bot

    if state["report_ready"] and not state["expanded"]:
        state["expanded"] = True
        st.rerun()

    with st.sidebar.expander("☀️ Solar Assistant", expanded=state["expanded"]):
        state["expanded"] = True

        # --- Input field at the top ---
        if "user_input" not in st.session_state:
            st.session_state.user_input = ""

        def on_enter():
            prompt = st.session_state.user_input.strip()
            if prompt:
                st.session_state.user_input = ""  # clear immediately
                handle_message(prompt)

        st.text_input(
            "Ask solar question:",
            key="user_input",
            label_visibility="visible",
            on_change=on_enter
        )

        # --- Messages container below input ---
        with st.container(height=300):
            # Reverse messages so newest appear first
            for msg in reversed(state["messages"][-12:]):
                if msg["role"] == "user":
                    st.markdown(f"**❔ :** {msg['content']}")
                else:
                    st.markdown(f"**☀️ :** {msg['content']}")
        
        # --- Quick buttons below messages ---
        col1, col2 = st.columns(2)
        if col1.button("Subsidies?", key="btn1", use_container_width=True):
            handle_message("Solar subsidies?")
        if col2.button("Costs?", key="btn2", use_container_width=True):
            handle_message("Costs?")


def handle_message(prompt):
    state = st.session_state.rag_bot
    if not prompt.strip(): return
    
    state["messages"].append({"role": "user", "content": prompt})
    
    try:
        client = Groq(api_key=GROQ_KEY)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=state["messages"],
            temperature=0.7, max_tokens=500
        )
        state["messages"].append({"role": "assistant", "content": resp.choices[0].message.content})
    except Exception as e:
        state["messages"].append({"role": "assistant", "content": f"Error: {str(e)[:150]}..."})
    
    st.session_state.input = ""  # Clear
    st.rerun()  # Single rerun
