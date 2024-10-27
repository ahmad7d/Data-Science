# data_loading.py
import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        """
        Initializes the DataLoader by loading the dataset.
        :param file_path: Path to the CSV file.
        """
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """
        Loads the data from the specified file path.
        :return: Loaded DataFrame
        """
        self.data = pd.read_csv(self.file_path)
        return self.data

    def inspect_data(self):
        """
        Inspects the first few rows and the structure of the dataset.
        """
        if self.data is not None:
            print(self.data.head())
            print(self.data.info())
        else:
            print("Data has not been loaded yet.")


if __name__ == "__main__":
    # Example usage
    loader = DataLoader('owid-co2-data.csv')
    co2_data = loader.load_data()
    loader.inspect_data()

    loader = DataLoader('fao_data.csv')
    co2_data2 = loader.load_data()

    loader = DataLoader('fao_data_without_energy.csv')
    co2_data3 = loader.load_data()


    print("------------------------------")
    print(co2_data.head())
    print(co2_data2.head())
    print(co2_data3.head())
    print("------------------------------")


    # from data_loading import DataLoader
    #
    # loader1 = DataLoader('owid-co2-data.csv')
    # loader2 = DataLoader('fao_data_without_energy.csv')
    # loader3 = DataLoader('fao_data.csv')
    #
    # print(loader1.data.head())
    # print(loader2.data.head())
    # print(loader3.data.head())
