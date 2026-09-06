import pandas as pd
import os
import re
from datetime import datetime

from base import DataPaths

import shutil

class FileExplorer:
    def __init__(self):
        pass

    def show_columns(self):
        for i,file in enumerate(os.listdir(DataPaths.bronze_path)):
            if file.endswith('.xlsx'):
                df = pd.read_excel(os.path.join(DataPaths.bronze_path, file))
                print(f"{i+1}. {file}")
                print(df.columns)
                print("-"*100)

    def show_data_types(self):
        for file in os.listdir(DataPaths.bronze_path):
            if file.endswith('.xlsx'):
                df = pd.read_excel(os.path.join(DataPaths.bronze_path, file))
                print(f"{file}")
                print(df.dtypes)
                print("-"*100)

if __name__ == "__main__":
    print("Starting the program...")
    print("Creating FileExplorer instance...")
    explorer = FileExplorer()
    print("Exploring data...")
    explorer.show_data_types()
    print("Program ended.")


    # raw_str = "Thu Feb 02 2017 15:43:04 GMT+0100 (Central European Standard Time)"

    # # 1. Remove the parenthetical timezone name
    # clean_str = raw_str.split(" (")[0]

    # # 2. Parse into a datetime object
    # dt = datetime.strptime(clean_str, "%a %b %d %Y %H:%M:%S GMT%z")

    # print(dt)