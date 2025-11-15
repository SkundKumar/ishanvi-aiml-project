import streamlit as st
import textwrap
import os

from ai import ensure_model, generate_response, words, classes


@st.cache_resource
def get_model():
    # If model.pth exists, load it; otherwise train a quick model (reduced epochs)
    model_path = 'model.pth'
    model = ensure_model(model_path=model_path, num_epochs=20, retrain=False, print_progress=False)
    return model


def main():
    st.title('Bennett University FAQ Chatbot')
    st.write('Ask about Bennett University admissions, courses, placements, hostels and more.')

    model = get_model()

    # Confidence threshold slider
    threshold = st.slider('Response confidence threshold', min_value=0.0, max_value=0.5, value=0.15, step=0.01)

    user_input = st.text_input('You:', '')
    if st.button('Send') and user_input.strip() != '':
        with st.spinner('Thinking...'):
            # generate_response now accepts a confidence_threshold parameter
            resp = generate_response(user_input.lower(), model, words, classes, confidence_threshold=threshold)
            st.markdown('**Bot:**')
            st.write(textwrap.fill(resp, width=80))


if __name__ == '__main__':
    main()
