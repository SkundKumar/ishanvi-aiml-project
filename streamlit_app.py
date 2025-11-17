import streamlit as st
import textwrap
import os
from datetime import datetime
import torch

# Assuming your 'ai.py' file is in the same directory
from ai import (
    ensure_model,
    generate_response,
    words,
    classes,
    preprocess_sentence,
    sentence_to_features,
)

# --- Page Config (Set this ONCE at the top) ---
# Use 'wide' layout for a more modern feel
# Set theme directly in config
st.set_page_config(page_title='Bennett Chatbot', layout='wide', initial_sidebar_state='collapsed')

# --- Model Loading ---
@st.cache_resource
def get_model():
    """Loads and caches the ML model."""
    model_path = 'model.pth'
    model = ensure_model(model_path=model_path, num_epochs=20, retrain=False, print_progress=False)
    return model

# --- Main App Logic ---
def main():
    st.title('Bennett University FAQ Chatbot')
    st.caption('Ask about admissions, courses, placements, hostels and more.')

    # Load the model once
    model = get_model()

    # Fixed confidence threshold
    threshold = 0.09

    # --- Modern Chat History (using st.session_state.messages) ---
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Add a "Clear Chat" button to the sidebar or a column
    with st.sidebar:
        st.title("Controls")
        if st.button('Clear Chat History'):
            st.session_state.messages = []
            st.session_state.last_tag = None
            st.rerun()

    # Display prior chat messages

    
    for message in st.session_state.messages:
        # Use Streamlit's built-in chat UI
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display metadata if it's from the assistant
            if message["role"] == "assistant" and "confidence" in message:
                st.caption(f"Confidence: {message['confidence']:.2f} • Tag: {message['tag']}")

    # --- Modern Chat Input (replaces the old form/CSS) ---
    # This input bar automatically sticks to the bottom
    if user_input := st.chat_input("Type your message..."):
        
        # 1. Add user message to UI and history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. Process the input
        user_text = user_input.strip()
        sentence_words = preprocess_sentence(user_text)
        sentence_words_in_vocab = [w for w in sentence_words if w in words]

        # Get previous tag for context
        prev_tag = st.session_state.get('last_tag')
        
        bot_resp = ""
        confidence = 0.0
        predicted_tag = "N/A"

        if len(sentence_words_in_vocab) == 0:
            # If no tokens recognized, use previous topic or fallback
            if prev_tag:
                bot_resp = generate_response(user_text, model, words, classes, confidence_threshold=threshold, prev_tag=prev_tag)
                predicted_tag = prev_tag # Use prev_tag as the tag
            else:
                bot_resp = "I'm sorry, but I don't understand. Can you please rephrase or provide more information?"
                predicted_tag = "fallback"
        else:
            # Generate new prediction
            features = sentence_to_features(sentence_words_in_vocab, words)
            with torch.no_grad():
                outputs = model(features)
            
            probs, idx = torch.max(outputs, dim=1)
            confidence = float(probs.item())
            predicted_tag = classes[idx.item()]

            bot_resp = generate_response(user_text, model, words, classes, confidence_threshold=threshold, prev_tag=prev_tag)
        
        # Save the last predicted tag for context
        st.session_state.last_tag = predicted_tag

        # 3. Add bot response to UI and history
        with st.chat_message("assistant"):
            st.markdown(bot_resp)
            st.caption(f"Confidence: {confidence:.2f} • Tag: {predicted_tag}")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_resp,
            "confidence": confidence,
            "tag": predicted_tag
        })

if __name__ == '__main__':
    main()