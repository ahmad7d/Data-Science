import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

class DataCleaner:
    def __init__(self, data):
        """
        Initializes the DataCleaner with the loaded dataset.
        :param data: DataFrame to clean
        """
        self.data = data
        self.cleaned_data = None

    def clean_data(self):
        """
        Cleans the dataset by handling missing values and retaining relevant columns.
        - We only drop columns with excessive missing values that are not necessary for the analysis.
        """
        # Drop columns that have too many missing values and are not critical to the analysis
        columns_to_drop = ['trade_co2', 'cement_co2', 'trade_co2_share']
        self.cleaned_data = self.data.drop(columns=columns_to_drop, axis=1)

        # Fill missing values in population and other relevant features
        self.cleaned_data['population'] = self.cleaned_data['population'].ffill()

        # Fill missing values in important CO2 columns using interpolation
        self.cleaned_data[['co2', 'co2_per_capita', 'coal_co2', 'oil_co2', 'gas_co2', 'flaring_co2', 'gdp']] = self.cleaned_data[
            ['co2', 'co2_per_capita', 'coal_co2', 'oil_co2', 'gas_co2', 'flaring_co2', 'gdp']
        ].interpolate()

        # Drop any remaining rows where 'co2' is still NaN (if interpolation failed)
        self.cleaned_data = self.cleaned_data.dropna(subset=['co2'])

        return self.cleaned_data

    def filter_countries(self, countries):
        """
        Filters the dataset to include only the specified countries.
        :param countries: List of countries
        :return: Filtered DataFrame
        """
        if self.cleaned_data is not None:
            return self.cleaned_data[self.cleaned_data['country'].isin(countries)]
        else:
            print("Data is not cleaned yet.")
            return None

    def filter_years(self, data, start_year, end_year):
        """
        Filters the dataset to include data within the specified year range.
        :param data: DataFrame to filter
        :param start_year: Start year
        :param end_year: End year
        :return: Filtered DataFrame
        """
        return data[(data['year'] >= start_year) & (data['year'] <= end_year)]


if __name__ == "__main__":
    # Example usage
    from data_loading import DataLoader

    loader = DataLoader('owid-co2-data.csv')
    co2_data = loader.load_data()

    # Clean the data using DataCleaner class
    cleaner = DataCleaner(co2_data)
    co2_cleaned_data = cleaner.clean_data()

    selected_countries = ['United States', 'China', 'India', 'Germany', 'Brazil', 'United Kingdom', 'Japan', 'Russia',
                          'Canada']
    co2_filtered_data = cleaner.filter_countries(selected_countries)
    co2_recent_data = cleaner.filter_years(co2_filtered_data, 1990, 2020)

    # Save the cleaned and filtered data for the next steps (e.g., PCA)
    co2_recent_data.to_csv('cleaned_filtered_co2_data.csv', index=False)
