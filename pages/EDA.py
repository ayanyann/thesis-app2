import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import tempfile
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from collections import Counter
import plotly.express as px

nltk.download('stopwords')
nltk.download('punkt')


def generate_word_distribution_histogram(data, abstract_note_column):
    # Ensure NLTK stopwords are available
    nltk_stopwords = set(stopwords.words('english'))
    # Combine all abstract notes into a single string
    all_notes_string = ' '.join(data[abstract_note_column].dropna().astype(str))
    # Tokenize the string into words
    words = nltk.word_tokenize(all_notes_string)
    # Filter out stopwords and non-alphabetic tokens
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in nltk_stopwords]
    # Count the occurrences of each word
    word_counts = Counter(filtered_words)
    # Convert the counter object to a DataFrame
    df_word_counts = pd.DataFrame(word_counts.items(), columns=['Word', 'Count']).sort_values(by='Count', ascending=False)
    # Generate the histogram using Plotly Express
    fig = px.bar(df_word_counts.head(30), x='Word', y='Count', title='Top 30 Word Frequencies')
    # Display the histogram
    st.plotly_chart(fig, use_container_width=True)

def generate_network_graph(data, author_column, hover_column=None):
    # Ensure the author column is treated as strings
    data[author_column] = data[author_column].astype(str)
    
    # Prepare the graph data
    authors_list = data[author_column].dropna().apply(lambda x: x.split("; ") if isinstance(x, str) else []).tolist()
    hover_info = {}
    if hover_column:
        # Ensure hover information is also treated as strings if it exists
        data[hover_column] = data[hover_column].astype(str)
        for index, row in data.iterrows():
            # Here, ensure row[author_column] is a string to avoid errors
            if isinstance(row[author_column], str):
                authors = row[author_column].split("; ")
                for author in authors:
                    hover_info[author] = row[hover_column]
            else:
                # Handle cases where row[author_column] is not a string
                authors = []

    edges = []
    for authors in authors_list:
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                edges.append((authors[i], authors[j]))  

    G = nx.Graph()
    G.add_edges_from(edges)

    pos = nx.spring_layout(G)
    # Calculate node sizes based on degree, applying a scale factor for visibility
    node_sizes = [len(list(G.neighbors(node))) * 0.5 for node in G.nodes()]  # Scale factor of 10 for example
    
    node_x, node_y, hover_texts = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        hover_text = f"{node}"
        if hover_column and node in hover_info:
            hover_text += f"<br>{hover_info[node]}"
        hover_texts.append(hover_text)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=hover_texts,
        marker=dict(
            showscale=True,
            color=[len(list(G.neighbors(node))) for node in G.nodes()],
            size= [1.5 + 1.5 * degree for degree in node_sizes],  # Use dynamic sizes based on degree
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(color='black', width=0.5)))

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0,l=0,r=0,t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    return fig

def generate_line_chart(data, year_column):
    # Count the number of papers per year
    year_counts = data[year_column].value_counts().sort_index()
    # Create the line chart
    fig = px.bar(x=year_counts.index, y=year_counts.values, labels={'x': 'Year', 'y': 'Number of Papers'})
    
    return fig


def generate_bar_chart(data, author_column, top_n=30):
    author_counts = data[author_column].str.split('; ').explode().value_counts().head(top_n)
    fig = px.bar(x=author_counts.index, y=author_counts.values, labels={'x': 'Author', 'y': 'Count'})
    
    return fig

# WordCloud generation function
def generate_wordcloud(data, tags_column):
    # Combine all tags into a single string
    all_tags_string = '; '.join(data[tags_column].dropna().astype(str))
    # Split the tags and combine into a single string again
    all_tags_list = all_tags_string.split('; ')
    all_tags = ' '.join(all_tags_list)
    
    # Generate the word cloud
    wordcloud = WordCloud(width = 800, height = 400, background_color ='white').generate(all_tags)
    
    # Display the word cloud using matplotlib
    plt.figure(figsize = (10, 5), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    
    # Use Streamlit to display matplotlib figure
    st.pyplot(plt)
def main():
    st.title("Exploratory Data Analysis of the Uploaded Dataset")

    if 'uploaded_file' in st.session_state:
        uploaded_file = st.session_state['uploaded_file']

        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name

        # Now read from the temporary file path
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(file_path)
        else:
            data = pd.read_excel(file_path)

        author_column = st.sidebar.selectbox("Select the author column", data.columns)
        year_column = st.sidebar.selectbox("Select the year column", data.columns)

        # New feature: Year filter
        unique_years = ["All"] + sorted(data[year_column].dropna().unique().tolist())
        selected_year = st.sidebar.selectbox("Filter by Year", unique_years)

        if selected_year != "All":
            data = data[data[year_column] == selected_year]

        # Display Network Graph
        hover_column = st.sidebar.selectbox("Select the column for hover information", ["None"] + list(data.columns))
        hover_column = None if hover_column == "None" else hover_column
        
        
        network_fig = generate_network_graph(data, author_column, hover_column)
        st.markdown('Author Networks')
        st.plotly_chart(network_fig, use_container_width=True)

        # Display the line chart and bar chart side by side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("Distribution of Papers Per Year")
            line_chart_fig = generate_line_chart(data, year_column)
            st.plotly_chart(line_chart_fig, use_container_width=True)

        with col2:
            st.markdown("Top Authors")
            bar_chart_fig = generate_bar_chart(data, author_column)
            st.plotly_chart(bar_chart_fig, use_container_width=True)
        
        if 'Automatic Tags' in data.columns:
                    st.markdown("### Wordcloud")
                    generate_wordcloud(data, 'Automatic Tags')
                    
        if 'Abstract Note' in data.columns:
            
            generate_word_distribution_histogram(data, 'Abstract Note')

        
    else:
        st.info("Please upload a file in the LDA Model section first.")
        

if __name__ == "__main__":
    main()