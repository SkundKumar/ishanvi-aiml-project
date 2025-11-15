import streamlit as st
import textwrap
import os

from ai import ensure_model, generate_response, words, classes

import streamlit as st
import textwrap
import os
from datetime import datetime
import torch

from ai import (
    ensure_model,
    generate_response,
    words,
    classes,
    preprocess_sentence,
    sentence_to_features,
)


st.set_page_config(page_title='Bennett Chatbot', layout='centered')


@st.cache_resource
def get_model():
    model_path = 'model.pth'
    model = ensure_model(model_path=model_path, num_epochs=20, retrain=False, print_progress=False)
    return model


def append_message(sender, message, **meta):
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({'sender': sender, 'message': message, 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), **meta})


def clear_history():
    st.session_state.history = []


def main():
    st.title('Bennett University FAQ Chatbot')
    st.caption('Ask about admissions, courses, placements, hostels and more.')

    # Top bar: load model and clear button
    col1, col2 = st.columns([8, 2])
    with col1:
        st.write('')
    with col2:
        if st.button('Clear Chat'):
            clear_history()

    with st.spinner('Loading model...'):
        model = get_model()

    # Fixed confidence threshold (no slider)
    threshold = 0.05

    # Inject CSS for dark background, high contrast bubbles, and fixed bottom input
    st.markdown(
        """
        <style>
        /* Page background */
        .stApp {
            background-color: #0b1220;
            color: #e6eef8;
        }
        /* Chat container spacing */
        .chat-container { padding-bottom: 120px; }

        /* User bubble */
        .user-bubble {
            background: linear-gradient(180deg,#0f766e,#064e3b);
            color: #e6eef8;
            padding: 12px;
            border-radius: 12px;
            display: inline-block;
            max-width: 80%;
        }

        /* Bot bubble */
        .bot-bubble {
            background: linear-gradient(180deg,#0b1730,#0b1220);
            color: #e6eef8;
            padding: 12px;
            border-radius: 12px;
            display: inline-block;
            max-width: 90%;
        }

        /* Fixed bottom input bar */
        .chat-input-bar {
            position: fixed;
            bottom: 12px;
            left: 50%;
            transform: translateX(-50%);
            width: 88%;
            background: transparent;
            z-index: 9999;
        }

        .chat-meta { font-size:11px; color:#98a0b3; margin-top:6px }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Chat history container (with extra bottom padding so fixed input doesn't overlap)
    chat_container = st.container()

    # Input form placed visually at the bottom using CSS wrapper
    with st.container():
        st.markdown("<div class='chat-input-bar'>", unsafe_allow_html=True)
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input('', placeholder='Type your message and press Enter', key='chat_input')
            submit = st.form_submit_button('Send')
        st.markdown("</div>", unsafe_allow_html=True)

    # Handle submission
    if submit and user_input and user_input.strip() != '':
        user_text = user_input.strip()
        append_message('user', user_text)
        # create features and compute confidence/tag
        sentence_words = preprocess_sentence(user_text)
        sentence_words_in_vocab = [w for w in sentence_words if w in words]

        # prev_tag from session (conversational context)
        prev_tag = st.session_state.get('last_tag') if 'last_tag' in st.session_state else None

        if len(sentence_words_in_vocab) == 0:
            # If no tokens recognized, try to continue previous topic when available
            if prev_tag:
                bot_resp = generate_response(user_text, model, words, classes, confidence_threshold=threshold, prev_tag=prev_tag)
                # use prev_tag as the tag shown
                append_message('bot', bot_resp, confidence=0.0, tag=prev_tag)
                st.session_state.last_tag = prev_tag
            else:
                # fallback immediately
                bot_resp = "I'm sorry, but I don't understand. Can you please rephrase or provide more information?"
                append_message('bot', bot_resp, confidence=0.0, tag='')
        else:
            features = sentence_to_features(sentence_words_in_vocab, words)
            with torch.no_grad():
                outputs = model(features)
            probs, idx = torch.max(outputs, dim=1)
            confidence = float(probs.item())
            predicted_tag = classes[idx.item()]

            # Generate response using the model and threshold, pass prev_tag for conversational context
            bot_resp = generate_response(user_text, model, words, classes, confidence_threshold=threshold, prev_tag=prev_tag)
            append_message('bot', bot_resp, confidence=confidence, tag=predicted_tag)
            # save last predicted tag to session state for follow-ups
            st.session_state.last_tag = predicted_tag

    # Render chat history
    with chat_container:
        if 'history' not in st.session_state or len(st.session_state.history) == 0:
            st.info('No messages yet. Ask something about Bennett University!')
        else:
            for msg in st.session_state.history:
                sender = msg.get('sender')
                text = msg.get('message')
                t = msg.get('time')
                if sender == 'user':
                        st.markdown(
                            f"<div style='text-align:right; margin:8px 0;'><div style='display:inline-block; background:linear-gradient(180deg,#0f766e,#064e3b); color:#e6eef8; padding:10px; border-radius:10px'>{text}</div><div style='font-size:11px; color:#98a0b3'>{t}</div></div>",
                            unsafe_allow_html=True,
                        )
                else:
                    confidence = msg.get('confidence', 0.0)
                    tag = msg.get('tag', '')
                    st.markdown(
                        f"<div style='text-align:left; margin:8px 0;'><div style='display:inline-block; background:linear-gradient(180deg,#0b1730,#0b1220); color:#e6eef8; padding:10px; border-radius:10px'>{text}<div style='font-size:11px; color:#98a0b3; margin-top:6px'>Confidence: {confidence:.2f} • Tag: {tag}</div></div><div style='font-size:11px; color:#98a0b3'>{t}</div></div>",
                        unsafe_allow_html=True,
                    )


if __name__ == '__main__':
    main()
