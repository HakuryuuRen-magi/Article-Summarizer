import streamlit as st
import requests



# Streamlit page setup
st.title("Article Summarizer")
st.write("Enter an article text and click 'Summarize' to generate a summary.")

# Input text area for the article
article_text = st.text_area("Paste your article text here:", height=300)

# Function to call the FastAPI backend
def get_summary(article_text):
    url = "http://localhost:8000/summarize"
    data = {"text": article_text}
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an error for bad HTTP status
        summary = response.json().get("summary", "No summary returned")
        return summary
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

# Button to trigger summarization
if st.button("Summarize"):
    if not article_text:
        st.warning("Please enter some text to summarize.")
    else:
        st.write("Summarizing...")
        summary = get_summary(article_text)
        st.success("Summary generated successfully!")
        st.write(summary)
