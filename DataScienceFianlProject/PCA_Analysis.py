import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.switch_backend('Agg')


class PCAAnalysis:
    def __init__(self, data):
        """
        Initializes the PCAAnalysis class with the dataset.
        :param data: DataFrame to analyze
        """
        self.data = data
        self.pca = None
        self.scaled_data = None
        self.principal_components = None

    def prepare_data(self):
        """
        Prepares the data for PCA by selecting relevant features and scaling.
        """
        # Select relevant features for PCA
        features = ['coal_co2', 'oil_co2', 'gas_co2', 'gdp', 'population']
        x = self.data[features].dropna()

        print(x)

        # Standardize the data
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(x)

        return self.scaled_data

    def apply_pca(self, n_components=2):
        """
        Applies PCA to the standardized data.
        :param n_components: Number of principal components to keep
        """
        self.pca = PCA(n_components=n_components)
        self.principal_components = self.pca.fit_transform(self.scaled_data)

        # Create a DataFrame with the principal components
        df_pca = pd.DataFrame(self.principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
        return df_pca

    def explained_variance_plot(self):
        """
        Plots the explained variance ratio by the principal components.
        """
        plt.figure(figsize=(8,6))
        plt.bar(range(1, len(self.pca.explained_variance_ratio_)+1), self.pca.explained_variance_ratio_)
        plt.xlabel('Principal Components')
        plt.ylabel('Variance Explained')
        plt.title('Explained Variance by Principal Components')
        plt.savefig("lalala1")

    def pca_scatter_plot(self, df_pca):
        """
        Creates a scatter plot of the first two principal components.
        """
        plt.figure(figsize=(10,7))
        plt.scatter(df_pca['PC1'], df_pca['PC2'])
        for i, country in enumerate(self.data['country']):
            plt.text(df_pca['PC1'][i], df_pca['PC2'][i], country)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Plot of Countries Based on CO2 Emissions and GDP')
        plt.savefig("lalala2")

if __name__ == "__main__":
    # Load the cleaned and filtered data
    co2_data = pd.read_csv('cleaned_filtered_co2_data.csv')

    # Initialize the PCAAnalysis class
    pca_analysis = PCAAnalysis(co2_data)

    # Prepare the data
    scaled_data = pca_analysis.prepare_data()

    # Apply PCA
    df_pca = pca_analysis.apply_pca()

    # Plot explained variance
    pca_analysis.explained_variance_plot()

    # Plot PCA scatter plot
    pca_analysis.pca_scatter_plot(df_pca)
