import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score  # Add silhouette score
plt.switch_backend('Agg')


class KMeansClustering:
    def __init__(self, data, n_clusters=3):
        """
        Initializes the KMeansClustering class with the dataset and number of clusters.
        :param data: DataFrame for clustering
        :param n_clusters: Number of clusters for K-Means
        """
        self.data = data
        self.n_clusters = n_clusters
        self.kmeans = None
        self.cluster_labels = None

    def scale_data(self, features):
        """
        Standardize the data before clustering.
        :param features: List of features to use for clustering
        :return: Scaled feature data
        """
        scaler = StandardScaler()
        return scaler.fit_transform(self.data[features])

    def apply_kmeans(self, scaled_data):
        """
        Apply K-Means clustering to the scaled data.
        :param scaled_data: Standardized feature data
        """
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.cluster_labels = self.kmeans.fit_predict(scaled_data)
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
        print(scaled_data)

        self.apply_kmeans(scaled_data)
        self.add_cluster_column()

        # Plot GDP vs CO2
        self.plot_clusters('gdp_per_capita', 'co2', title="GDP per Capita vs CO2 Emissions")
        self.print_countries_in_clusters()
        self.calculate_silhouette_score(scaled_data)  # Call silhouette score calculation

    def plot_2_co2_per_capita_vs_sectoral(self):
        """
        Plot for CO2 Emissions Per Capita and Sectoral Emissions using:
        'co2_per_capita', 'gdp_per_capita', 'coal_co2', 'oil_co2', 'gas_co2'
        """
        features = ['co2_per_capita', 'gdp_per_capita', 'coal_co2', 'oil_co2', 'gas_co2']
        scaled_data = self.scale_data(features)
        print(scaled_data)

        self.apply_kmeans(scaled_data)
        self.add_cluster_column()

        # Plot CO2 per Capita vs GDP per Capita
        self.plot_clusters('gdp_per_capita', 'co2_per_capita', title="CO2 per Capita vs GDP per Capita")
        self.print_countries_in_clusters()
        self.calculate_silhouette_score(scaled_data)  # Call silhouette score calculation

    def plot_3_global_contributions(self):
        """
        Plot for Total CO2 Emissions and Global Contributions using:
        'co2', 'primary_energy_consumption', 'share_global_co2', 'coal_co2', 'oil_co2'
        """
        features = ['co2', 'primary_energy_consumption', 'share_global_co2', 'coal_co2', 'oil_co2']
        scaled_data = self.scale_data(features)
        print(scaled_data)
        self.apply_kmeans(scaled_data)
        self.add_cluster_column()

        # Plot Primary Energy Consumption vs CO2
        self.plot_clusters('primary_energy_consumption', 'co2', title="Primary Energy Consumption vs CO2 Emissions")
        self.print_countries_in_clusters()
        self.calculate_silhouette_score(scaled_data)  # Call silhouette score calculation

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
        plt.savefig(f'K-Means_Clustering_on_{feature_x}_vs_{feature_y}')
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
        plt.title('Number of Countries in Each Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Number of Countries')
        plt.xticks(rotation=0)
        plt.savefig("kmeans_cluster_distribution")
        plt.close()

    def calculate_silhouette_score(self, scaled_data):
        """
        Calculate and print the silhouette score to evaluate the quality of the clustering.
        """
        score = silhouette_score(scaled_data, self.cluster_labels)
        print(f"Silhouette Score for K-Means: {score}")
        return score

    def plot_elbow_method(self, features):
        """
        Plot the Elbow Method to determine the optimal number of clusters.
        :param features: List of features to use for clustering
        """
        scaled_data = self.scale_data(features)
        wcss = []
        for i in range(1, 11):  # Testing between 1 and 10 clusters
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(scaled_data)
            wcss.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 11), wcss, marker='o')
        plt.title("Elbow Method for Optimal K")
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig('kmeans_elbow_method.png')
        plt.close()


if __name__ == "__main__":
    # Load the cleaned and filtered data
    co2_data = pd.read_csv('cleaned_filtered_co2_data.csv')

    # Initialize the KMeansClustering class
    kmeans_clustering = KMeansClustering(co2_data, n_clusters=3)

    # Calculate GDP per Capita and add to dataset
    kmeans_clustering.calculate_gdp_per_capita()

    # Plot the Elbow method (optional) to determine the optimal number of clusters
    features = ['co2', 'gdp', 'oil_co2', 'gas_co2', 'gdp_per_capita']
    kmeans_clustering.plot_elbow_method(features)

    # Plot 1: CO2 Emissions and Economic Factors
    kmeans_clustering.plot_1_co2_vs_economic_factors()

    # Plot 2: CO2 Emissions Per Capita and Sectoral Emissions
    kmeans_clustering.plot_2_co2_per_capita_vs_sectoral()

    # Plot 3: Total CO2 Emissions and Global Contributions
    kmeans_clustering.plot_3_global_contributions()
