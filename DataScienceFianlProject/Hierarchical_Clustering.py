# HierarchicalClustering.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
plt.switch_backend('Agg')


class HierarchicalClustering:
    def __init__(self, data, n_clusters=3, linkage_method='ward'):
        """
        Initializes the HierarchicalClustering class with the dataset, number of clusters, and linkage method.
        :param data: DataFrame for clustering
        :param n_clusters: Number of clusters for Hierarchical Clustering
        :param linkage_method: Linkage method for clustering ('ward', 'complete', 'average', 'single')
        """
        self.data = data
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.cluster_labels = None
        self.model = None

    def scale_data(self, features):
        """
        Standardize the data before clustering.
        :param features: List of features to use for clustering
        :return: Scaled feature data
        """
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data[features])
        return scaled_data

    def apply_hierarchical_clustering(self, scaled_data):
        """
        Apply Hierarchical Agglomerative Clustering to the scaled data.
        :param scaled_data: Standardized feature data
        """
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage_method)
        self.cluster_labels = self.model.fit_predict(scaled_data)
        return self.cluster_labels

    def add_cluster_column(self):
        """
        Add a cluster label column to the dataset.
        """
        self.data['cluster'] = self.cluster_labels

    def calculate_gdp_per_capita(self):
        """
        Adds a new column 'gdp_per_capita' by dividing GDP by population.
        """
        self.data['gdp_per_capita'] = self.data['gdp'] / self.data['population']

    def plot_1_co2_vs_economic_factors(self):
        """
        Plot for CO2 Emissions and Economic Factors using:
        'co2', 'gdp', 'oil_co2', 'gas_co2', 'gdp_per_capita'
        """
        features = ['co2', 'gdp', 'oil_co2', 'gas_co2', 'gdp_per_capita']
        scaled_data = self.scale_data(features)

        self.apply_hierarchical_clustering(scaled_data)
        self.add_cluster_column()

        # Plot GDP vs CO2
        self.plot_clusters('gdp_per_capita', 'co2', title="Hierarchical Clustering: GDP per Capita vs CO2 Emissions")
        self.print_countries_in_clusters()

        # Plot Dendrogram
        self.plot_dendrogram(scaled_data, title="Dendrogram for CO2 vs Economic Factors")

    def plot_2_co2_per_capita_vs_sectoral(self):
        """
        Plot for CO2 Emissions Per Capita and Sectoral Emissions using:
        'co2_per_capita', 'gdp_per_capita', 'coal_co2', 'oil_co2', 'gas_co2'
        """
        features = ['co2_per_capita', 'gdp_per_capita', 'coal_co2', 'oil_co2', 'gas_co2']
        scaled_data = self.scale_data(features)

        self.apply_hierarchical_clustering(scaled_data)
        self.add_cluster_column()

        # Plot CO2 per Capita vs GDP per Capita
        self.plot_clusters('gdp_per_capita', 'co2_per_capita', title="Hierarchical Clustering: CO2 per Capita vs GDP per Capita")
        self.print_countries_in_clusters()

        # Plot Dendrogram
        self.plot_dendrogram(scaled_data, title="Dendrogram for CO2 per Capita vs Sectoral Emissions")

    def plot_3_global_contributions(self):
        """
        Plot for Total CO2 Emissions and Global Contributions using:
        'co2', 'primary_energy_consumption', 'share_global_co2', 'coal_co2', 'oil_co2'
        """
        features = ['co2', 'primary_energy_consumption', 'share_global_co2', 'coal_co2', 'oil_co2']
        scaled_data = self.scale_data(features)

        self.apply_hierarchical_clustering(scaled_data)
        self.add_cluster_column()

        # Plot Primary Energy Consumption vs CO2
        self.plot_clusters('primary_energy_consumption', 'co2', title="Hierarchical Clustering: Primary Energy Consumption vs CO2 Emissions")
        self.print_countries_in_clusters()

        # Plot Dendrogram
        self.plot_dendrogram(scaled_data, title="Dendrogram for Global Contributions")

    def plot_clusters(self, feature_x, feature_y, title):
        """
        General plot function to visualize clustering results.
        :param feature_x: Feature for the x-axis
        :param feature_y: Feature for the y-axis
        :param title: Title of the plot
        """
        plt.figure(figsize=(10, 7))
        plt.scatter(self.data[feature_x], self.data[feature_y], c=self.cluster_labels, cmap='viridis')
        plt.title(title)
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.colorbar(label='Cluster')
        plt.savefig(f'Hierarchical_Clustering_on_{feature_x}_vs_{feature_y}.png')
        plt.close()

    def print_countries_in_clusters(self):
        """
        Prints the countries in each cluster and displays a bar plot of the number of countries in each cluster.
        """
        cluster_counts = self.data['cluster'].value_counts().sort_index()
        clusters = self.data.groupby('cluster')['country'].unique()

        print("\nCountries in Each Cluster:")
        for cluster_id, countries in clusters.items():
            print(f"Cluster {cluster_id}: {', '.join(countries)}")

        # Plot a bar chart showing the number of countries in each cluster
        plt.figure(figsize=(8, 5))
        cluster_counts.plot(kind='bar', color='skyblue')
        plt.title('Number of Countries in Each Cluster (Hierarchical Clustering)')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Countries')
        plt.xticks(rotation=0)
        plt.savefig("hierarchical_cluster_distribution.png")
        plt.close()

    def plot_dendrogram(self, scaled_data, title):
        """
        Plots a truncated dendrogram for the hierarchical clustering.
        :param scaled_data: Standardized feature data used for clustering
        :param title: Title of the dendrogram plot
        """
        # Generate the linkage matrix
        linked = linkage(scaled_data, method=self.linkage_method)

        plt.figure(figsize=(15, 10))  # Increase figure size for better readability
        dendrogram(linked,
                   labels=self.data['country'].values,
                   orientation='top',
                   distance_sort='descending',
                   show_leaf_counts=True,
                   leaf_rotation=90,  # Rotate labels to make them horizontal
                   leaf_font_size=6,  # Reduce label font size for more clarity
                   truncate_mode='lastp',  # Show only the last p merged clusters
                   p=30  # Show only the last 30 clusters
                   )
        plt.title(title)
        plt.xlabel('Countries')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.savefig(f"Dendrogram_{title.replace(' ', '_')}.png")
        plt.close()


if __name__ == "__main__":
    # Load the cleaned and filtered data
    co2_data = pd.read_csv('cleaned_filtered_co2_data.csv')

    # Initialize the HierarchicalClustering class
    hierarchical_clustering = HierarchicalClustering(co2_data, n_clusters=3, linkage_method='ward')

    # Calculate GDP per Capita and add to dataset
    hierarchical_clustering.calculate_gdp_per_capita()

    # Plot 1: CO2 Emissions and Economic Factors
    hierarchical_clustering.plot_1_co2_vs_economic_factors()

    # Plot 2: CO2 Emissions Per Capita and Sectoral Emissions
    hierarchical_clustering.plot_2_co2_per_capita_vs_sectoral()

    # Plot 3: Total CO2 Emissions and Global Contributions
    hierarchical_clustering.plot_3_global_contributions()
