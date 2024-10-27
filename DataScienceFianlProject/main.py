import pandas as pd

# Load the CO2 dataset
co2_data = pd.read_csv('owid-co2-data.csv')

# Checking the structure of the dataset
# print(co2_data.head())
# print(co2_data.info())



# Selecting relevant columns
columns_to_keep = [
    'country', 'year', 'co2', 'co2_per_capita', 'coal_co2', 'oil_co2', 'gas_co2',
    'cement_co2', 'flaring_co2', 'trade_co2', 'population'
]

# Creating a new dataframe with selected columns
co2_data_filtered = co2_data[columns_to_keep]

# Checking the first few rows again to confirm
# print(co2_data_filtered.head())


# Check for missing values
missing_data = co2_data_filtered.isnull().sum() / len(co2_data_filtered) * 100
# print(missing_data)
# print(co2_data_filtered.info())


# Filter data for the years from 1990 onward
# co2_data_recent = co2_data_filtered[co2_data_filtered['year'] >= 1990]

# Check the shape of the dataset after filtering
# print(co2_data_recent.info())


# Drop columns with more than 50% missing data
co2_data_cleaned = co2_data_filtered.drop(['trade_co2', 'cement_co2'], axis=1)


# Forward fill for population (filling missing values with previous valid value)
co2_data_cleaned['population'] = co2_data_cleaned['population'].fillna(method='ffill')

# Interpolate for CO2-related columns (time-series interpolation)
co2_data_cleaned[['co2', 'co2_per_capita', 'coal_co2', 'oil_co2', 'gas_co2', 'flaring_co2']] = co2_data_cleaned[['co2', 'co2_per_capita', 'coal_co2', 'oil_co2', 'gas_co2', 'flaring_co2']].interpolate()



# Check if any missing values are left
print(co2_data_cleaned.isnull().sum())
