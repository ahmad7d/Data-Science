import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def load_data(file_path):
    """
    Loads a CSV file into a pandas DataFrame.
    Reads CO₂ emissions data and sector data from CSV files.
    :param file_path: CSV file path.
    return: Pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    """
    Cleans the DataFrame by dropping rows with missing values.
    :param df: Pandas DataFrame
    :return df: Cleaned DataFrame
    """
    df.dropna(inplace=True)
    return df


def filter_countries(df, countries):
    """
    Filters the DataFrame to include only the specified countries.
    :param df: Pandas DataFrame
    :param countries: List of countries
    return: filtered Pandas DataFrame on countries
    """
    return df[df['country'].isin(countries)]


def filter_years(df, start_year, end_year):
    """
    Filters the DataFrame to include data within the specified year range.
    :param df: Pandas DataFrame
    :param start_year: Start year
    :param end_year: End year
    :return df: filtered Pandas DataFrame on years
    """
    return df[(df['year'] >= start_year) & (df['year'] <= end_year)]


def plot_co2_trends(df, save_path="co2_emissions_trends.png"):
    """
    Plots the CO₂ emissions over time for each country.
    Output: A line chart showing emission trends saved as an image file.
    :param df: Pandas DataFrame
    :param save_path: Path to save the image
    """
    plt.figure(figsize=(10, 6))
    for country in df['country'].unique():
        country_data = df[df['country'] == country]
        plt.plot(country_data['year'], country_data['co2'], label=country)

    plt.title('CO2 Emissions Trends (1990-2020)')
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions (Million Tons)')
    plt.legend(title="country")
    plt.savefig(save_path)
    plt.close()
    print(f"CO2 Emissions trends saved to {save_path}")


def plot_sector_contributions(df, country, save_path="sector_contributions.png"):
    """
    Creates a pie chart showing the percentage contribution of each sector to the country's total CO₂ emissions.
    :param df: Pandas DataFrame
    :param country: Country
    :param save_path: Path to save the image
    """
    country_df = df[df['country'] == country]
    sector_data = country_df.groupby('Item').sum()['co2_emissions']

    # Filter out negative values
    sector_data = sector_data[sector_data >= 0]

    plt.figure(figsize=(8, 6))
    sector_data.plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f'{country} CO2 Emissions by Sector (1990-2020)')
    plt.ylabel('')  # Remove y-axis label for pie chart
    plt.savefig(save_path)
    plt.close()
    print(f"Sector contribution chart saved to {save_path}")


def plot_policy_impact(country):
    """
    Visualizes the impact of environmental policies by marking significant policy years on the CO₂ emissions trend line.
    Policy Impact Assessment
    Marking Kyoto Protocol and Paris Agreement on the CO2 trend plot
    :param country: Country
    """
    country_data = co2_data_filtered[co2_data_filtered['country'] == country]
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='year', y='co2_emissions', data=country_data, marker="o", label=country)

    # Highlight Kyoto Protocol (1997) and Paris Agreement (2015)
    plt.axvline(x=1997, color='r', linestyle='--', label='Kyoto Protocol (1997)')
    plt.axvline(x=2015, color='g', linestyle='--', label='Paris Agreement (2015)')

    plt.title(f'CO2 Emissions and Policy Impact for {country} (1990-2020)')
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions (million tons)')
    plt.legend()
    plt.grid(True)
    plt.savefig('germany_policy_impact.png')


def polynomial_regression(df, degree=3):
    """
    Fits a polynomial regression model to capture non-linear trends in CO₂ emissions over time.
    Data Preparation: Uses 'year' as the feature and 'co2_emissions' as the target.
    Polynomial Features: Transforms the original features into higher-degree terms.
    Model Training: Splits data into training and testing sets and fits the model.
    Evaluation: Calculates Mean Squared Error (MSE) and R-squared (R²) to assess performance.
    Visualization: Plots actual vs. predicted emissions.
    :param df: Pandas DataFrame
    :param degree: polynomial degree
    :return model: Polynomial regression model
    """
    # Feature selection
    features = ['year']
    target = 'co2_emissions'

    # Prepare the data
    X = df[features]
    y = df[target]

    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # Initialize Linear Regression model
    poly_model = LinearRegression()

    # Train the model
    poly_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = poly_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Polynomial Regression (degree={degree}) Results:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Plot Actual vs Predicted CO2 Emissions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # 45-degree line
    plt.title(f'Polynomial Regression (degree={degree}): Actual vs Predicted CO2 Emissions')
    plt.xlabel('Actual CO2 Emissions')
    plt.ylabel('Predicted CO2 Emissions')
    plt.savefig('poly_actual_vs_predicted.png')
    plt.close()
    print("Prediction plot saved as 'poly_actual_vs_predicted.png'")

    return poly_model


def train_linear_regression(df, feature_col, target_col):
    """
    Fits a simple linear regression model to predict CO₂ emissions based on the year.
    Data Preparation: Extracts features and target variable.
    Model Training: Splits data and fits the model.
    Evaluation: Computes MSE and R².
    :param df: Pandas DataFrame
    :param feature_col: Feature column
    :param target_col: Target column
    :return: Linear regression model
    """
    X = df[[feature_col]].values  # Year as feature
    y = df[target_col].values  # CO2 emissions as target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')

    return model


def predict_future_emissions(model, future_years, country, save_path="future_predictions.png"):
    """
    Uses the trained linear regression model to forecast future CO₂ emissions.
    Output: A line chart showing predicted emissions saved as an image file.
    :param model: Linear regression model
    :param future_years: List of future years
    :param country: Country
    :param save_path: Path to save the image
    """
    future_years_array = np.array(future_years).reshape(-1, 1)
    predicted_emissions = model.predict(future_years_array)

    plt.figure(figsize=(8, 6))
    plt.plot(future_years, predicted_emissions, marker='o', label=f'Predicted CO2 Emissions for {country}')
    plt.title(f'Predicted CO2 Emissions for {country} (2021-2040)')
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions (Million Tons)')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Future predictions saved to {save_path}")


def random_forest_prediction(df):
    """
    Fits a Random Forest regression model to predict CO₂ emissions.
    Captures non-linear relationships and interactions between variables.
    Model Training: Uses ensemble learning with multiple decision trees.
    Evaluation: Assesses performance using MSE and R².
    Visualization: Plots actual vs. predicted emissions.
    :param df: Pandas DataFrame
    :return: Random forest model
    """
    features = ['year']
    target = 'co2_emissions'

    # Prepare the data
    X = df[features]
    y = df[target]

    # Split the dataset into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model on the training set
    rf_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Random Forest Regression Results:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    # Plot Actual vs Predicted CO2 Emissions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    plt.title('Random Forest: Actual vs Predicted CO2 Emissions')
    plt.xlabel('Actual CO2 Emissions')
    plt.ylabel('Predicted CO2 Emissions')
    plt.savefig('rf_actual_vs_predicted.png')
    plt.close()
    print("Prediction plot saved as 'rf_actual_vs_predicted.png'")

    return rf_model


if __name__ == "__main__":
    # Load Data
    co2_data = load_data('owid-co2-data.csv')
    sector_data = load_data('fao_data.csv')
    fao_data_without_energy_sector = pd.read_csv('fao_data_without_energy.csv')  # FAO data (without energy sector)

    # Preprocess Data
    co2_data = clean_data(co2_data)
    sector_data = clean_data(sector_data)
    sector_data_without = clean_data(fao_data_without_energy_sector)



    # Filter Countries and Years
    selected_countries = ['United States', 'China', 'India', 'Germany', 'United Kingdom', 'Brazil', 'South Africa',
                          'Nigeria', 'Mexico']
    co2_data_filtered = filter_countries(co2_data, selected_countries)
    co2_data_filtered = filter_years(co2_data_filtered, 1990, 2020)


    # Plot CO2 Trends for the selected countries
    plot_co2_trends(co2_data_filtered, save_path="co2_emissions_trends.png")

    # Analyzing sector contributions for the United States (as an example)
    plot_sector_contributions(sector_data, "United States of America", 'us_sector_contributions.png')

    # Analyzing sector contributions for the United States without the energy sector.
    plot_sector_contributions(sector_data_without, "United States of America",
                              'us_sector_contributions_without_energy.png')

    # Comparing the CO2 emissions between China and UK
    china_uk = filter_countries(co2_data_filtered, ['China', 'United Kingdom'])
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=china_uk, x='year', y='co2_emissions', hue='country')
    plt.title('CO2 Emissions Comparison: China vs UK (1990-2020)')
    plt.ylabel('CO2 Emissions (Million Tons)')
    plt.xlabel('year')
    plt.savefig('china_uk_comparison.png')
    plt.close()

    # Example: Policy impact on Germany's CO2 emissions
    plot_policy_impact('Germany')

    # Train Linear Regression model for CO2 emissions prediction
    model = train_linear_regression(co2_data_filtered, feature_col='year', target_col='co2_emissions')

    # Predict future emissions (for example: for 2021 to 2040)
    future_years = list(range(2021, 2041))
    predict_future_emissions(model, future_years, country='United States', save_path="future_predictions_usa.png")

    # Random Forest Model
    random_forest_model = random_forest_prediction(co2_data_filtered)

    # Polynomial Regression Model
    polynomial_model = polynomial_regression(co2_data_filtered, degree=3)
