import streamlit as st

def initialize_chat():
    """Sets up the chat history and toggle state in session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "How can I help with your solar project?"}
        ]
    if "chat_expanded" not in st.session_state:
        st.session_state.chat_expanded = False

def render_chat_interface():
    """Renders a floating AI assistant icon in the bottom-right corner."""

    # CSS for the floating container
    st.markdown("""
        <style>
        /* Fixed position container for AI button */
        .ai-float-container {
            position: fixed;
            bottom: 25px;
            right: 25px;
            z-index: 9999;
        }

        /* The circular AI button */
        .ai-float-btn {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transition: transform 0.2s, box-shadow 0.2s;
            text-decoration: none;
            color: white;
        }

        .ai-float-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }

        .ai-float-label {
            text-align: center;
            margin-top: 5px;
            font-size: 11px;
            color: #666;
            font-weight: 500;
        }

        /* Expanded chat panel */
        .ai-chat-panel {
            position: fixed;
            bottom: 25px;
            right: 25px;
            width: 320px;
            z-index: 9999;
            background: white;
            border-radius: 12px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.15);
            overflow: hidden;
        }

        .ai-chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .ai-chat-header h4 {
            margin: 0;
            font-size: 16px;
        }

        .ai-chat-close {
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .ai-chat-close:hover {
            background: rgba(255,255,255,0.3);
        }

        .ai-chat-body {
            padding: 20px;
        }

        .coming-soon-icon {
            font-size: 48px;
            text-align: center;
            margin-bottom: 15px;
        }
        </style>
    """, unsafe_allow_html=True)

    if not st.session_state.chat_expanded:
        # Show floating icon button
        st.markdown("""
            <div class="ai-float-container">
                <div class="ai-float-btn" onclick="window.location.href='?chat=open'" title="AI Assistant">
                    🤖
                </div>
                <div class="ai-float-label">AI Assistant</div>
            </div>
        """, unsafe_allow_html=True)

        # Check URL param to open chat
        if st.query_params.get("chat") == "open":
            st.session_state.chat_expanded = True
            st.query_params.clear()
            st.rerun()
    else:
        # Show expanded chat panel
        st.markdown("""
            <div class="ai-chat-panel">
                <div class="ai-chat-header">
                    <h4>🤖 SolarSight Assistant</h4>
                    <button class="ai-chat-close" onclick="window.location.href='?chat=close'" title="Close">✕</button>
                </div>
                <div class="ai-chat-body">
                    <div class="coming-soon-icon">🚧</div>
                    <p style="text-align: center; color: #666; margin-bottom: 15px;">
                        <strong>Coming Soon</strong>
                    </p>
                    <p style="color: #888; font-size: 14px; line-height: 1.5;">
                        AI-powered solar advisor is under development. Soon you'll be able to:
                    </p>
                    <ul style="color: #888; font-size: 13px; line-height: 1.8;">
                        <li>Ask questions about your analysis</li>
                        <li>Get personalized recommendations</li>
                        <li>Understand your energy savings</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Check URL param to close chat
        if st.query_params.get("chat") == "close":
            st.session_state.chat_expanded = False
            st.query_params.clear()
            st.rerun()