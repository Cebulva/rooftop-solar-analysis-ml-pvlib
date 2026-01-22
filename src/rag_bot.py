import streamlit as st

def initialize_chat():
    """Sets up the chat history in session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How can I help with your solar project?"}
        ]

def render_chat_interface():
    """Renders the fixed chat window at the bottom."""
    # 1. CSS for Fixed Position
    st.markdown("""
        <style>
        .stChatFloatingInputContainer {
            background-color: rgba(0,0,0,0);
        }
        div[data-testid="stVerticalBlock"] > div:has(div.fixed-chat) {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 350px;
            z-index: 1000;
            background: white;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # 2. Chat UI Container
    with st.container():
        st.markdown('<div class="fixed-chat">', unsafe_allow_html=True)
        st.caption("🤖 SolarSight Assistant")
        
        # Display chat messages (limited height for the floating window)
        chat_placeholder = st.empty()
        with chat_placeholder.container():
            for msg in st.session_state.messages[-3:]: # Show last 3 for space
                st.chat_message(msg["role"]).write(msg["content"])
        
        # User Input
        if prompt := st.chat_input("Ask a question...", key="rag_input"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Placeholder for RAG logic
            response = f"I'm analyzing the data for Step {st.session_state.step}. Ask me about solar potential!"
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)