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
import numpy as np
import io  
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

def download_nltk_data():
    try:
        nltk.data.find('stopwords')
        nltk.data.find('punkt')
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')

def load_custom_stopwords(file_path):
    try:
        with open(file_path, 'r') as file:
            custom_stopwords = file.read().splitlines()
        return custom_stopwords
    except FileNotFoundError:
        st.error("Custom stopwords file not found.")
        return []

def preprocess_text(text, preprocess_steps):
    # Update to load custom stopwords
    custom_stopwords_path = 'custom_stopwords_for_research_articles.txt'  # Update the path if necessary
    custom_stopwords = load_custom_stopwords(custom_stopwords_path)
    
    nltk.corpus.wordnet.ensure_loaded()
    
    if pd.isnull(text) or text.strip() == "":
        return ""

    # Convert to lowercase
    if "Convert to lowercase" in preprocess_steps:
        text = text.lower()
    
    # Tokenize
    word_tokens = word_tokenize(text)

    # Remove stopwords
    if "Remove stopwords" in preprocess_steps:
        stop_words = set(stopwords.words('english')) | set(custom_stopwords)
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

def generate_word_clouds(lda_model, num_topics):
    # Calculate the number of rows needed, assuming 2 word clouds per row
    num_rows = (num_topics + 1) // 2
    for i in range(num_rows):
        cols = st.columns(2)  # Create two columns for each row
        for j in range(2):
            topic_index = i * 2 + j
            if topic_index < num_topics:
                with cols[j]:  # Use the column as the context
                    plt.figure(figsize=(2, 2))  # Smaller figure size
                    wc = WordCloud(background_color='white', width=200, height=200)
                    wc.fit_words(dict(lda_model.show_topic(topic_index, 200)))
                    plt.imshow(wc, interpolation='bilinear')
                    plt.axis("off")
                    plt.title(f"Theme #{topic_index+1}")
                    st.pyplot(plt)  # Use Streamlit's function to display the plot

def main():
    st.title("Identifying Information Systems Themes")
    st.write("by: Mark Ryan Uy, Nathaniel Jay Maña")
    
    # Session state for file upload
    uploaded_file = st.file_uploader("Choose a file to upload, supported formats are only. CSV and .XLSX", type=["csv", "xlsx", "json"])
    if uploaded_file is not None:
        st.session_state['uploaded_file'] = uploaded_file
    
    if 'uploaded_file' in st.session_state:
        # Read the file into a BytesIO buffer
        file_buffer = io.BytesIO(st.session_state.uploaded_file.getvalue())

        # Process file based on its extension
        if st.session_state.uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(file_buffer)  # Use file_buffer instead of uploaded_file directly
        elif st.session_state.uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(file_buffer)  # Use file_buffer here as well
        elif st.session_state.uploaded_file.name.endswith('.json'):
            df = pd.read_json(file_buffer)  # And here
        else:
            st.error("Unsupported file type.")
            return
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Preview")
            st.write(df.head(3))
        
        # Sidebar - Column selection
        st.sidebar.subheader("Select the Abstract column for analysis,")
        selected_column = st.sidebar.selectbox("Select column", df.columns)
        # Filter column selection
        
        # Sidebar - Filter Selection
        st.sidebar.subheader("Filter the Dataset")
        filter_column = st.sidebar.selectbox("Select a column to filter by:", df.columns)
        
        # Sidebar - Filter Values Selection with an 'All' option
        all_values = df[filter_column].unique().tolist()
        all_values.sort()  # Sort values for better user experience
        selected_values = st.sidebar.multiselect("Select values to filter by (select 'All' for all values):", ['All'] + all_values)
        
        # Check if 'All' is selected in the filter values
        if 'All' in selected_values:
            # If 'All' is selected, we use all unique values for the filter
            selected_values = all_values
        
        # Apply filters to dataframe if filter values are selected
        if selected_values:
            df = df[df[filter_column].isin(selected_values)]
        
        
        # Display the filtered DataFrame
        #st.sidebar.subheader("Filtered Dataset")
        #st.sidebar.write(filtered_df)
        
        # Display the selected column from the filtered DataFrame
        with col2:
            st.subheader("Filtered Column")
            st.write(df[selected_column].head(3))
        
        
        
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
        df["processed_text"] = df[selected_column].apply(lambda x: preprocess_text(x, preprocess_steps))
        # Display the processed text in the sidebar
        # st.sidebar.write(filtered_df["processed_text"].head(3))
        
        # Tokenize the processed text from the filtered DataFrame
        tokenized_text = [word_tokenize(text) for text in df["processed_text"]]
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(tokenized_text)
        corpus = [dictionary.doc2bow(text) for text in tokenized_text]
        
        
        # LDA Topic Modeling
        st.subheader("LDA Topic Modeling")
        
       # Sidebar - Extended LDA parameters
        st.sidebar.subheader("LDA Parameters - Extended")
       # Existing parameters
        num_topics = st.sidebar.number_input("Number of Themes", min_value=1, max_value=100, value=5)
        num_passes = st.sidebar.number_input("Number of Passes", min_value=1, max_value=1000, value=10)
        num_iterations = st.sidebar.number_input("Number of Iterations", min_value=1, max_value=1000, value=100)
        num_words = st.sidebar.number_input("Number of Words per Theme", min_value=1, max_value=100, value=10)
        # New parameters
        alpha = st.sidebar.select_slider("Alpha (Document-Topic Density)", options=['symmetric', 'asymmetric', 'auto'] + list(np.arange(0.01, 1, 0.01)))
        beta = st.sidebar.select_slider("Beta (Topic-Word Density)", options=['symmetric', 'auto'] + list(np.arange(0.01, 1, 0.01)))
        random_state = st.sidebar.number_input("Random State", min_value=0, max_value=1000, value=42)
        update_every = st.sidebar.number_input("Update Every", min_value=1, max_value=5, value=1)
        chunksize = st.sidebar.number_input("Chunksize", min_value=1, max_value=5000, value=2000)
        
        # Progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Model is Running, please wait...")

        # Dummy progress update (replace with actual model training progress)
        for percent_complete in range(100):
            progress_bar.progress(percent_complete + 1)
        
        lda_model = models.LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=num_passes,
            iterations=num_iterations,
            alpha=alpha,
            eta=beta,
            random_state=random_state,
            chunksize=chunksize
        )

        # Clear progress bar and status text
        progress_bar.empty()
        status_text.empty()

        
        # Step 1: Get the dominant topic for each document along with the percentage contribution
        dominant_topics = []
        for doc_id, doc in enumerate(corpus):
            topic_percs = lda_model.get_document_topics(doc)
            dominant_topic = sorted(topic_percs, key=lambda x: x[1], reverse=True)[0]
            dominant_topics.append((doc_id, dominant_topic[0], dominant_topic[1]))

        # Step 2: Create a DataFrame with Document ID, Dominant Topic, and Percentage Contribution
        doc_topic_df = pd.DataFrame(dominant_topics, columns=['Document ID', 'Dominant Theme', 'Percentage Contribution'])

        # Assuming 'df' is your original DataFrame and it includes a 'Title' column
        df_with_topics = df.join(doc_topic_df.set_index('Document ID'))
        # Optional: Sort by Dominant Topic or any other column as needed
        df_with_topics = df_with_topics.sort_values(by='Dominant Theme')
        # Step 3: Select relevant columns to display in the table, for example, Title and Dominant Topic
        display_df = df_with_topics[['Title', 'Dominant Theme', 'Percentage Contribution']]
        
        # Group by 'Dominant Topic' and apply a lambda function to each group to sort and select top 10
        top_papers_per_topic = df_with_topics.groupby('Dominant Theme').apply(
            lambda x: x.sort_values('Percentage Contribution', ascending=False).head(50)
        ).reset_index(drop=True)

                
        cal1, cal2 = st.columns(2)
        # Visualize the topic model
        with st.subheader("Model Visualization using LDAvis"):
            vis = gensimvis.prepare(lda_model, corpus, dictionary)
            html_string = pyLDAvis.prepared_data_to_html(vis)
            st.components.v1.html(html_string, width=None, height=800)

        # Evaluate the topic model
        with cal1, st.expander("Model Evaluation"):
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
        
        with cal2, st.expander("Generated Themes"):
        # Prepare data for the table
            themes_data = []
            for idx, topic in lda_model.print_topics(num_words=num_words):
            # Extracting words and their corresponding weights
                words = topic.split('+')
                # Remove weights and keep just the words for simplicity
                words = [re.search(r'\"(.+?)\"', word).group(1) if re.search(r'\"(.+?)\"', word) else word for word in words]
                themes_data.append({'Theme': idx, 'Words': ', '.join(words)})
    
            # Convert the list of dictionaries to a DataFrame
            themes_df = pd.DataFrame(themes_data)

            # Ensure that the table displays at least 3 rows
            min_rows = max(2, len(themes_df))  # Set minimum rows to 3 or the number of themes, whichever is larger
            st.table(themes_df.head(min_rows))
            
    
        # Call the function to generate word cloud visualizations
        with st.expander('Wordcloud'):
            generate_word_clouds(lda_model, num_topics)
                
        # Display the table in Streamlit
        with st.expander("Top Papers per Theme"):
            # You might want to sort by 'Dominant Topic' to have a nice ordered table
            top_papers_per_topic = top_papers_per_topic.sort_values(by=['Dominant Theme', 'Percentage Contribution'], ascending=[True, False])
        
            st.table(top_papers_per_topic[['Title', 'Dominant Theme']])
            
        # Line Chart

        topic_year_distribution = df_with_topics.groupby(['Publication Year', 'Dominant Theme']).size().unstack(fill_value=0)
        topic_year_long_df = topic_year_distribution.reset_index().melt(id_vars=['Publication Year'], var_name='Dominant Theme', value_name='Document Count')
                
        # Assuming 'topic_year_long_df' is your DataFrame prepared for plotting
        fig = px.line(topic_year_long_df, x='Publication Year', y='Document Count', color='Dominant Theme',
                    title='Theme Distribution Over Years',
                    markers=True, # Adds markers to the line
                    line_shape='linear') # Ensures the line is linear; adjust if necessary

        # Customizing the chart for a more aesthetic appeal
        fig.update_layout(
            plot_bgcolor='white', # Sets background color to white
            xaxis_title='Publication Year',
            yaxis_title='Number of Documents',
            legend_title='Dominant Theme',
            font=dict(
                family="Arial, sans-serif",
                size=12,
                color="RebeccaPurple"
            )
        )

        # Customizing axes and grid lines for better readability
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='lightgrey')
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='lightgrey')

        # Display the chart in the Streamlit app
        st.plotly_chart(fig, use_container_width=True)







        
if __name__ == "__main__":
    download_nltk_data()
    main()