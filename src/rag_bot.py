import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
import random

load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("LLM_MODEL")

SYSTEM_PROMPT = """
You are a helpful solar assistant for private homeowners in Germany.

Rules:
- Focus exclusively on residential rooftop solar installations for private houses.
- Use report data ONLY for numbers and factual values; never modify them.
- You MAY use general knowledge to explain, interpret, and analyze what the report numbers mean.
- All answers must be relevant to Germany.
- Keep answers clear, well structured, and easy to read.
- Avoid tables (width is limited); prefer short paragraphs or bullet points.
- If information is missing from the report, say so politely.
- Be friendly and supportive.

Length & guidance:
- Initial answers should be very short (~30-60 words), just key info.
- Only expand in detail if user asks via a follow-up question.

Follow-up logic:
- Always return exactly ONE short, topic-style follow-up question (1–2 words) in this exact format:
  Next question: <question>
- Do NOT include this question in the main answer text; it will only be shown as a button.
- Suggestions should be actionable, specific, and relevant (e.g., "PV basics", "Installation", "Cost reduction").
- Do NOT duplicate content from the main answer.
- If a report is NOT ready, suggest general educational follow-ups.
- If a report IS ready, suggest report-specific analytical follow-ups.
"""

# Fallback questions if LLM doesn't generate one
FALLBACK_QUESTIONS = [
    "Costs",
    "Installation",
    "Savings",
    "Subsidies",
    "Maintenance",
    "ROI",
    "Battery storage",
    "Grid connection",
    "Energy yield",
    "Roof requirements"
]

def initialize_chat():
    if "rag_bot" not in st.session_state:
        st.session_state.rag_bot = {
            "messages": [{"role": "assistant", "content": "How can I assist you today?"}],
            "expanded": False,
            "report_ready": False,
            "next_question": "PV basics",  # default first question
            "used_fallbacks": set()  # track which fallbacks we've used
        }

def get_fallback_question(state):
    """Get a random fallback question that hasn't been used recently"""
    available = [q for q in FALLBACK_QUESTIONS if q not in state.get("used_fallbacks", set())]
    
    # Reset if we've used all fallbacks
    if not available:
        state["used_fallbacks"] = set()
        available = FALLBACK_QUESTIONS
    
    question = random.choice(available)
    state["used_fallbacks"].add(question)
    return question

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
            # Add custom CSS for compact dividers
            st.markdown("""
                <style>
                .stMarkdown hr {
                    margin-top: 0.5rem !important;
                    margin-bottom: 0.5rem !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
            messages_to_show = state["messages"][-12:]
            # Reverse to show newest first
            messages_to_show = list(reversed(messages_to_show))
            
            for i, msg in enumerate(messages_to_show):
                if msg["role"] == "user":
                    st.markdown(f"**❔ :** {msg['content']}")
                else:
                    st.markdown(f"**☀️ :** {msg['content']}")
                    
                    # Show divider after each assistant response, except the last one (which is now the oldest)
                    if i < len(messages_to_show) - 1:
                        st.markdown("---")

        # --- Single follow-up button below messages ---
        # Always ensure we have a next_question
        if not state.get("next_question"):
            state["next_question"] = get_fallback_question(state)
        
        if state.get("next_question"):
            # Use dynamic key based on question content to force re-render
            button_key = f"followup_{hash(state['next_question'])}"
            if st.button(f"➡️ {state['next_question']}", key=button_key, use_container_width=True):
                # Capture the question before it gets cleared
                question_to_send = state["next_question"]
                # Clear it immediately to prevent re-clicks
                state["next_question"] = None
                # Send follow-up to the chat
                handle_message(question_to_send)
                # Force rerun to update UI
                st.rerun()

def handle_message(prompt):
    state = st.session_state.rag_bot
    if not prompt.strip():
        return

    state["messages"].append({"role": "user", "content": prompt})

    try:
        client = Groq(api_key=GROQ_KEY)
        messages_with_system = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]

        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages_with_system,
            temperature=0.7,
            max_tokens=500
        )

        content = resp.choices[0].message.content.strip()

        # --- Extract follow-up question ---
        next_q = None
        if "Next question:" in content:
            try:
                next_q = content.split("Next question:")[1].strip().split("\n")[0]  # only first line
                # Clean up any extra formatting
                next_q = next_q.replace("*", "").replace("**", "").strip()
                if next_q:  # only use if not empty
                    # Reset fallback tracking when LLM provides a question
                    if "used_fallbacks" in state:
                        state["used_fallbacks"].clear()
            except:
                next_q = None

        # Append assistant message without the follow-up
        main_content = content.split("Next question:")[0].strip()
        state["messages"].append({"role": "assistant", "content": main_content})
        
        # Set the next question AFTER appending the message
        # If LLM didn't provide one, fallback will be assigned in render_chat_interface
        state["next_question"] = next_q

    except Exception as e:
        state["messages"].append({"role": "assistant", "content": f"Error: {str(e)[:150]}..."})
        # Ensure we have a fallback question even on error
        state["next_question"] = None

# Initialize chat on load
initialize_chat()
render_chat_interface()