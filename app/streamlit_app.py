"""
Streamlit UI for the Amazon Review Sentiment Analyser.

How to run
----------
1. Start the FastAPI server first:
       uvicorn api.main:app --reload --port 8000

2. In a separate terminal, from the project root:
       streamlit run app/streamlit_app.py
"""

import streamlit as st
import requests

API_URL = 'http://localhost:8080/predict'

SENTIMENT_CONFIG = {
    'positive': {'emoji': '😊', 'color': '#28a745', 'label': 'Positive'},
    'neutral':  {'emoji': '😐', 'color': '#ffc107', 'label': 'Neutral'},
    'negative': {'emoji': '😞', 'color': '#dc3545', 'label': 'Negative'},
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title='Review Sentiment Analyser',
    page_icon='🛍️',
    layout='centered',
)

st.title('🛍️ Amazon Review Sentiment Analyser')
st.markdown(
    'Enter a product review and a star rating to predict whether the '
    'sentiment is **positive**, **neutral**, or **negative**.'
)

st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form('predict_form'):
    review_text = st.text_area(
        'Review text',
        height=150,
        placeholder='e.g. "Great product, exactly what I needed. Fast shipping too!"',
    )
    review_score = st.slider('Star rating (1 = worst, 5 = best)', 1, 5, value=3)
    submitted = st.form_submit_button('Predict Sentiment', use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    if not review_text.strip():
        st.warning('Please enter a review before predicting.')
    else:
        with st.spinner('Calling the API...'):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        'review_text': review_text,
                        'review_score': float(review_score),
                    },
                    timeout=10,
                )
                response.raise_for_status()
                result = response.json()
                sentiment = result['sentiment']
                cfg = SENTIMENT_CONFIG.get(
                    sentiment, {'emoji': '❓', 'color': '#6c757d', 'label': sentiment}
                )

                st.markdown(
                    f"<h2 style='color:{cfg['color']};'>"
                    f"{cfg['emoji']} Sentiment: {cfg['label']}"
                    f"</h2>",
                    unsafe_allow_html=True,
                )

                with st.expander('Details'):
                    st.json(result)

            except requests.exceptions.ConnectionError:
                st.error(
                    'Cannot connect to the API. '
                    'Make sure the FastAPI server is running:\n\n'
                    '`uvicorn api.main:app --reload --port 8000`'
                )
            except requests.exceptions.HTTPError as exc:
                st.error(f'API error: {exc.response.status_code} — {exc.response.text}')
            except Exception as exc:
                st.error(f'Unexpected error: {exc}')

st.divider()
st.caption('Powered by scikit-learn · FastAPI · Streamlit')
