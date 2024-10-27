import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from networkx.algorithms.community import girvan_newman

class GirvanNewmanClustering:
    def __init__(self, data, features=None):
        """
        Initialize the GirvanNewmanClustering class.
        :param data: DataFrame for clustering
        :param features: List of features to use for building the graph
        """
        self.data = data
        self.features = features
        self.graph = None
        self.communities = None

    def build_graph(self):
        """
        Build a graph from the dataset based on similarities in the selected features.
        """
        self.graph = nx.Graph()

        # Add nodes (countries)
        for index, row in self.data.iterrows():
            self.graph.add_node(row['country'], label=row['country'])

        # Add edges based on similarity between features
        for i, row1 in self.data.iterrows():
            for j, row2 in self.data.iterrows():
                if i != j:
                    # Use Euclidean distance between selected features to define edge weights
                    distance = sum((row1[feature] - row2[feature]) ** 2 for feature in self.features) ** 0.5
                    if distance < some_threshold:  # Define a threshold for similarity
                        self.graph.add_edge(row1['country'], row2['country'], weight=1 / (distance + 1))

    def apply_girvan_newman(self):
        """
        Apply the Girvan-Newman algorithm to detect communities.
        """
        # Using NetworkX's implementation of the Girvan-Newman algorithm
        self.communities = next(girvan_newman(self.graph))

    def plot_communities(self):
        """
        Plot the graph with detected communities highlighted.
        """
        # Assign each node to a community
        community_map = {}
        for idx, community in enumerate(self.communities):
            for node in community:
                community_map[node] = idx

        # Create color map based on communities
        colors = [community_map[node] for node in self.graph.nodes]

        # Plot graph with nodes colored by their community
        plt.figure(figsize=(10, 7))
        nx.draw(self.graph, with_labels=True, node_color=colors, cmap=plt.cm.rainbow, node_size=500, font_size=10)
        plt.title("Girvan-Newman Community Detection")
        plt.savefig('girvan_newman_communities.png')
        plt.show()

    def print_communities(self):
        """
        Print the detected communities.
        """
        for idx, community in enumerate(self.communities):
            print(f"Community {idx + 1}: {', '.join(community)}")


if __name__ == "__main__":
    # Load your cleaned dataset
    co2_data = pd.read_csv('cleaned_filtered_co2_data.csv')

    # Initialize Girvan-Newman Clustering
    features = ['gdp_per_capita', 'co2', 'oil_co2', 'gas_co2']  # You can modify these based on your analysis
    gn_clustering = GirvanNewmanClustering(co2_data, features=features)

    # Build the graph based on similarity of countries using selected features
    gn_clustering.build_graph()

    # Apply Girvan-Newman algorithm
    gn_clustering.apply_girvan_newman()

    # Plot and print the detected communities
    gn_clustering.plot_communities()
    gn_clustering.print_communities()
