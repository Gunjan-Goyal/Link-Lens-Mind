# Link-Lens-Mind
Link-Lens-Mind is a lightweight, Streamlit-based research assistant that uses an LLM (Large Language Model) to extract relevant information from online articles. Simply input URLs of web pages, and ask questions related to their content — the app will semantically search the text and return accurate answers.

It is built using LangChain, FAISS, and Google Generative AI embeddings.

## Demo
[Visit the App](https://link-lens-mind.streamlit.app/)

## Features
- Accepts one or more URLs of articles/blogs.
- Splits and embeds article content into a searchable vector store.
- Uses Google’s Generative AI Embeddings + LangChain RetrievalQA.
- Answers user questions based on the article content.
- Reduces hallucination risk by grounding answers in actual data.

## Installation
- Clone the repo
<pre> ```bash git clone https://github.com/Gunjan-Goyal/Link-Lens-Mind.git
cd Link-Lens-Mind``` </pre>
- Set up environment
<pre> ```bash pip install -r requirements.txt``` </pre>
- Add your API_key in .env.example file in the root directory:
<pre> ```bash GOOGLE_API_KEY=your_google_generative_ai_key``` </pre>
- Run the app
<pre> ```bash streamlit run main.py``` </pre>
