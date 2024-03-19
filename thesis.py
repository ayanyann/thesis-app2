import streamlit as st
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text(text, preprocess_steps):
    if pd.isnull(text) or text.strip() == "":
        return ""

    # Remove punctuation
    if "Remove punctuation" in preprocess_steps:
        text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers
    if "Remove numbers" in preprocess_steps:
        text = re.sub(r'\d+', '', text)

    # Remove special characters
    if "Remove special characters" in preprocess_steps:
        text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Convert to lowercase
    if "Convert to lowercase" in preprocess_steps:
        text = text.lower()

    # Remove stopwords
    if "Remove stopwords" in preprocess_steps:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_tokens = [w for w in word_tokens if w not in stop_words]
        text = ' '.join(filtered_tokens)

    # Remove short tokens
    if "Remove short tokens" in preprocess_steps:
        word_tokens = word_tokenize(text)
        filtered_tokens = [w for w in word_tokens if len(w) > 2]
        text = ' '.join(filtered_tokens)

    # Tokenize and lemmatize
    if "Tokenize and lemmatize" in preprocess_steps:
        lemmatizer = WordNetLemmatizer()
        word_tokens = word_tokenize(text)
        lemmatized_tokens = [lemmatizer.lemmatize(w) for w in word_tokens]
        text = ' '.join(lemmatized_tokens)

    return text



def main():
    st.title("Identifying Information Systems Themes")
    st.write("by: Mark Ryan Uy, Nathaniel Jay Maña -- under supervision of Dr. Adrian P. Galido Ph.D")



    # File upload
    uploaded_file = st.file_uploader("Choose a file to upload, must be csv, excel, and json", type=["csv", "xlsx", "json"])

    if uploaded_file is not None:
        # Check file type and read accordingly
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return
        # Display the first 5 rows of the dataset
        st.subheader("Dataset Preview")
        st.write(df.head(3))

        # Sidebar - Column selection
        st.sidebar.subheader("Select a column for Analysis, a text column would be perferred")
        selected_column = st.sidebar.selectbox("Select a text column", df.columns)

        # Sidebar - Filter column selection
        st.sidebar.subheader("filter by year ")
        filter_column = st.sidebar.selectbox("Select a column to filter by", df.columns)

        # Sidebar - Filter value selection
        st.sidebar.subheader("pick a year")
        filter_values = st.sidebar.multiselect("select values", df[filter_column].unique())

        # Filter the DataFrame based on the selected filter column and values
        filtered_df = df[df[filter_column].isin(filter_values)]

        # Display the filtered DataFrame
        st.subheader("Filtered Dataset")
        st.write(filtered_df)

        # Display the selected column from the filtered DataFrame
        st.sidebar.subheader("Filtered Column")
        st.sidebar.write(filtered_df[selected_column].head(3))

        # Sidebar - Preprocessing steps
        st.sidebar.subheader("Preprocessing")
        preprocess_steps = st.sidebar.multiselect("Select preprocessing", [
            "Remove punctuation",
            "Remove numbers",
            "Remove special characters",
            "Convert to lowercase",
            "Remove stopwords",
            "Remove short tokens",
            "Tokenize and lemmatize"
        ], default=["Remove punctuation", "Convert to lowercase"])

        # Apply preprocessing and create new column in the filtered DataFrame
        filtered_df["processed_text"] = filtered_df[selected_column].apply(lambda x: preprocess_text(x, preprocess_steps))

        # Display the processed text in the sidebar
        st.sidebar.subheader("Processed Text")
        st.sidebar.write(filtered_df["processed_text"].head(3))

        # Tokenize the processed text from the filtered DataFrame
        tokenized_text = [word_tokenize(text) for text in filtered_df["processed_text"]]

        # Create dictionary and corpus
        dictionary = corpora.Dictionary(tokenized_text)
        corpus = [dictionary.doc2bow(text) for text in tokenized_text]
        
        # LDA Topic Modeling
        st.subheader("LDA Topic Modeling")

        # Sidebar - LDA parameters
        st.sidebar.subheader("LDA Parameters")
        num_topics = st.sidebar.number_input("Number of Topics", min_value=1, max_value=100, value=5)
        num_passes = st.sidebar.number_input("Number of Passes", min_value=1, max_value=100, value=10)
        num_iterations = st.sidebar.number_input("Number of Iterations", min_value=1, max_value=1000, value=100)
        num_words = st.sidebar.number_input("Number of Words per Topic", min_value=1, max_value=20, value=10)

        # Train LDA model
        lda_model = models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                                        passes=num_passes, iterations=num_iterations)

        # Display topic-word distributions
        st.subheader("Topic-Word Distributions")
        for idx, topic in lda_model.print_topics(num_words=num_words):
            st.write("Topic {}: {}".format(idx, topic))

        # Evaluate the topic model
        st.subheader("Topic Model Evaluation")

        # Compute perplexity
        perplexity = lda_model.log_perplexity(corpus)
        st.write("Perplexity: {:.2f}".format(perplexity))

        # Compute coherence scores
        coherence_model = CoherenceModel(model=lda_model, texts=tokenized_text, dictionary=dictionary, coherence='c_v')
        coherence_model2 = CoherenceModel(model=lda_model, texts=tokenized_text, dictionary=dictionary, coherence='c_uci')
        coherence_model3 = CoherenceModel(model=lda_model, texts=tokenized_text, dictionary=dictionary, coherence='c_npmi')
        coherence_score = coherence_model.get_coherence()
        coherence_score2 = coherence_model2.get_coherence()
        coherence_score3 = coherence_model3.get_coherence()
        st.write("Coherence Score (c_v): {:.2f}".format(coherence_score))
        st.write("Coherence Score (c_cuci): {:.2f}".format(coherence_score2))
        st.write("Coherence Score (c_npmi): {:.2f}".format(coherence_score3))

        # Visualize the topic model
        st.subheader("Topic Model Visualization")
        vis = gensimvis.prepare(lda_model, corpus, dictionary)
        html_string = pyLDAvis.prepared_data_to_html(vis)
        st.components.v1.html(html_string, width=1400, height=800)

if __name__ == "__main__":
    main()