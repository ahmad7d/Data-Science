
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

plt.switch_backend('Agg')
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from data_loading import DataLoader  # Import the DataLoader class
from data_cleaning import DataCleaner  # Import the DataCleaner class




class DataAnalyzer:
    def __init__(self, data):
        """
        Initializes the DataAnalyzer with the cleaned dataset.
        :param data: Cleaned DataFrame
        """
        self.data = data

    def plot_co2_trends(self, countries, save_path="co2_emissions_trends.png"):
        """
        Plots CO2 emissions over time for the selected countries.
        :param countries: List of countries to include in the plot
        :param save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))

        for country in countries:
            country_data = self.data[self.data['country'] == country]
            plt.plot(country_data['year'], country_data['co2'], label=country)

        plt.title('CO2 Emissions Trends (1990-2020)')
        plt.xlabel('Year')
        plt.ylabel('CO2 Emissions (Million Tons)')
        plt.legend(title="Country")
        plt.savefig(save_path)
        plt.close()
        print(f"CO2 Emissions trends saved to {save_path}")

    def train_linear_regression(self, feature_col='year', target_col='co2'):
        """
        Trains a simple linear regression model to predict CO₂ emissions.
        :param feature_col: Feature column (default is 'year')
        :param target_col: Target column (default is 'co2')
        :return: Trained linear regression model
        """
        X = self.data[[feature_col]].values
        y = self.data[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f'Mean Squared Error: {mse}')
        print(f'R-squared: {r2}')

        return model

    def train_linear_regression_for_country(self, country, feature_cols, target_col='co2'):
        """
        Trains a linear regression model for a specific country using multiple features.
        :param country: The name of the country.
        :param feature_cols: List of feature columns.
        :param target_col: The target column (default is 'co2').
        :return: Trained linear regression model.
        """
        # Filter data for the specific country
        country_data = self.data[self.data['country'] == country]

        # Print the filtered country data to debug
        print(country_data.head())

        # Ensure we only drop rows where either any feature in feature_cols or target_col is NaN
        available_features = [col for col in feature_cols if col in country_data.columns]

        if len(available_features) == 0:
            print(f"No valid feature columns found for {country}.")
            return None

        country_data = country_data.dropna(subset=available_features + [target_col])

        # Extract the features (X) and target (y)
        X = country_data[available_features].values  # Multiple features as input
        y = country_data[target_col].values  # CO2 emissions as target

        if len(X) == 0 or len(y) == 0:
            print(f"Insufficient data for {country}")
            return None

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions for evaluation
        y_pred = model.predict(X_test)

        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"{country} Linear Regression Results:")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")

        return model

    def predict_future_emissions(self, model, future_years, country, save_path="future_predictions.png"):
        """
        Uses the trained country-specific linear regression model to forecast future CO₂ emissions.
        :param model: Trained linear regression model
        :param future_years: List of future years for prediction
        :param country: Country for which prediction is made
        :param save_path: Path to save the prediction plot
        """
        if model is None:
            print(f"No model found for {country}")
            return

        future_years_array = np.array(future_years).reshape(-1, 1)
        predicted_emissions = model.predict(future_years_array)

        # Plotting the future predictions
        plt.figure(figsize=(8, 6))
        plt.plot(future_years, predicted_emissions, marker='o', label=f'Predicted CO2 Emissions for {country}')
        plt.title(f'Predicted CO2 Emissions for {country} (2021-2040)')
        plt.xlabel('Year')
        plt.ylabel('CO2 Emissions (Million Tons)')
        plt.legend()
        plt.savefig(save_path)
        plt.close()
        print(f"Future predictions saved to {save_path}")

    def polynomial_regression(self, degree=3):
        """
        Trains a polynomial regression model to capture non-linear trends.
        :param degree: Degree of the polynomial (default is 3)
        :return: Trained polynomial regression model
        """
        X = self.data[['year']]
        y = self.data['co2']

        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

        poly_model = LinearRegression()
        poly_model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = poly_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Polynomial Regression (degree={degree}) Results:")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")

        return poly_model

    def random_forest_prediction(self):
        """
        Trains a Random Forest regression model to predict CO₂ emissions.
        :return: Trained Random Forest model
        """
        X = self.data[['year']]
        y = self.data['co2']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Random Forest Regression Results:")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")

        return rf_model





if __name__ == "__main__":
    # Step 1: Load data using the DataLoader class
    data_loader = DataLoader(file_path='owid-co2-data.csv')
    raw_co2_data = data_loader.load_data()

    # Step 2: Clean data using the DataCleaner class
    data_cleaner = DataCleaner(raw_co2_data)
    cleaned_co2_data = data_cleaner.clean_data()

    # Step 3: Analyze data with the DataAnalyzer class
    analyzer = DataAnalyzer(cleaned_co2_data)

    # List of countries for which predictions will be made
    countries = ['United States', 'China', 'India', 'Germany', 'Brazil']

    future_years = list(range(2021, 2041))

    # Features that we want to use (but may not be available for all countries)
    features = ['cement_co2', 'coal_co2', 'gas_co2', 'oil_co2', 'energy_per_capita', 'primary_energy_consumption']
    # Train model for each country
    for country in countries:
        model = analyzer.train_linear_regression_for_country(country, features, target_col='co2')
        if model:
            analyzer.predict_future_emissions(model, future_years, country,
                                              save_path=f"future_predictions_{country.lower()}.png")
