import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

def generate_network_graph(data, author_column, hover_column=None):
    # Prepare the graph data
    authors_list = data[author_column].dropna().apply(lambda x: x.split("; ")).tolist()
    hover_info = {}
    if hover_column:
        for index, row in data.iterrows():
            authors = row[author_column].split("; ")
            for author in authors:
                hover_info[author] = row[hover_column]

    edges = []
    for authors in authors_list:
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                edges.append((authors[i], authors[j]))

    G = nx.Graph()
    G.add_edges_from(edges)

    pos = nx.spring_layout(G)
    
    node_x, node_y, hover_texts = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        hover_text = f"{node}"
        if hover_column and node in hover_info:
            hover_text += f"<br>{hover_info[node]}"
        hover_texts.append(hover_text)
    
    
    # Before creating node_trace, calculate node degrees and scale sizes
    node_degrees = [len(list(G.neighbors(node))) for node in G.nodes()]
# Example scaling: 10px base size + 5px per connection, adjust as needed
    node_sizes = [2 + 1.5 * degree for degree in node_degrees]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=hover_texts,
        marker=dict(
            showscale=True,
            color=node_degrees,  # Color intensity based on degree
            size=node_sizes,  # Use the calculated sizes here
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

def main():
    st.title("Author Network Graph")
    
    uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Read the uploaded file
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        
        author_column = st.sidebar.selectbox("Select the author column", data.columns)
        
        # Dataset filtering feature
        filter_column = st.sidebar.selectbox("Filter dataset by", ["None"] + list(data.columns))
        if filter_column != "None":
            filter_values = data[filter_column].unique()
            selected_filter_value = st.sidebar.selectbox(f"Select a value to filter by {filter_column}", filter_values)
            data = data[data[filter_column] == selected_filter_value]
        
        hover_column = st.sidebar.selectbox("Select the column for hover information", ["None"] + list(data.columns))
        hover_column = None if hover_column == "None" else hover_column
        
        fig = generate_network_graph(data, author_column, hover_column)
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Please upload a CSV or Excel file.")

if __name__ == "__main__":
    main()
