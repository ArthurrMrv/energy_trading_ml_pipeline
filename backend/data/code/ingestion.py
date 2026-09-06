from base import DataPaths, DataTemplates

import os
import re

import pandas as pd
import polars as pl

import shutil

# 2020 flows renamed these; archive files still use the old headers.
_FLOW_COLUMNS = {
    "Submitted By": "FromAreaCode",
    "Border with": "ToAreaCode",
    "SB Member": "FromAreaMemberType",
    "BW Member": "ToAreaMemberType",
}


def _align_flow_columns(df: pl.DataFrame) -> pl.DataFrame:
    for old, new in _FLOW_COLUMNS.items():
        if old not in df.columns:
            continue
        if new in df.columns:
            df = df.with_columns(pl.coalesce(pl.col(new), pl.col(old)).alias(new)).drop(old)
        else:
            df = df.rename({old: new})
    return df

class Ingestion:

    def bronze_to_silver(self):
        silver_dfs = {}
        silver_years = {}

        # entso-e files in bronze path
        bronze_files = [f for f in os.listdir(os.path.join(DataPaths.bronze_path, 'entsoe')) if f.endswith('.xlsx')]
        bronze_files_path = [os.path.join(DataPaths.bronze_path, 'entsoe', file) for file in bronze_files]
        for i, (file, bronze_file_path) in enumerate(zip(bronze_files, bronze_files_path), 1):
            print(f"\rProcessing {file} ({i}/{len(bronze_files)})", end="", flush=True)

            name = re.sub(r'-\d{4}(?:-\d{4})?$', '', os.path.splitext(file)[0])
            path = os.path.join(DataPaths.silver_path, f'{name}.parquet')

            if name not in silver_dfs:
                if os.path.isfile(path):
                    df = pl.read_parquet(path)
                    if name == DataTemplates.PHYSICAL_ENERGY_POWER_FLOWS and any(
                        c in df.columns for c in _FLOW_COLUMNS
                    ):
                        df = _align_flow_columns(df)
                        df.write_parquet(path)
                    silver_dfs[name] = df
                    silver_years[name] = set(df['Year'].unique().to_list())
                else:
                    silver_dfs[name] = None
                    silver_years[name] = set()

            match = re.search(r'-(\d{4})(?:-(\d{4}))?\.xlsx$', file)
            if not match:
                continue

            start, end = int(match.group(1)), int(match.group(2) or match.group(1))
            missing = set(range(start, end + 1)) - silver_years[name]
            if not missing:
                continue

            bronze = pd.concat(pd.read_excel(bronze_file_path, sheet_name=None).values(), ignore_index=True)
            if name == DataTemplates.PHYSICAL_ENERGY_POWER_FLOWS:
                bronze = bronze.rename(columns=_FLOW_COLUMNS)
            if name == DataTemplates.MONTHLY_HOURLY_LOAD_VALUES:
                bronze['Year'] = bronze['DateUTC'].dt.year
            elif 'Year' not in bronze.columns:
                raise ValueError(f'Year column not found in {file}')

            bronze = bronze[bronze['Year'].isin(missing)].copy()
            if bronze.empty:
                continue

            for col in bronze.columns:
                if bronze[col].dtype == object:
                    bronze[col] = bronze[col].astype('string')
            incoming = pl.from_pandas(bronze)
            silver_dfs[name] = (
                incoming
                if silver_dfs[name] is None
                else pl.concat([silver_dfs[name], incoming], how='diagonal_relaxed')
            )
            silver_years[name].update(incoming['Year'].unique().to_list())
            silver_dfs[name].write_parquet(path)

        # ember files in bronze path
        # Just load the file and save it into silver as parquet
        # if csv path doesnt exist raise error 
        if not os.path.exists(os.path.join(DataPaths.bronze_path, 'ember', f'{DataTemplates.EUROPEAN_WHOLESALE_ELECTRICITY_PRICE_DATA_DAILY}.csv')):
            raise FileNotFoundError(f"File {os.path.join(DataPaths.bronze_path, 'ember', f'{DataTemplates.EUROPEAN_WHOLESALE_ELECTRICITY_PRICE_DATA_DAILY}.csv')} not found")

        bronze_file = os.path.join(DataPaths.bronze_path, 'ember', f'{DataTemplates.EUROPEAN_WHOLESALE_ELECTRICITY_PRICE_DATA_DAILY}.csv')
        silver_file = os.path.join(DataPaths.silver_path, f'{DataTemplates.EUROPEAN_WHOLESALE_ELECTRICITY_PRICE_DATA_DAILY}.parquet')
        bronze = pd.read_csv(bronze_file)
        silver = pl.from_pandas(bronze)
        silver.write_parquet(silver_file)
        
        print()

    def silver_to_gold(self):
        # PACEHOLDER
        # data is clean and therefor doesn't need to be transformed
        for file in os.listdir(DataPaths.silver_path):
            print(f"\rCopying {file}", end="", flush=True)
            shutil.copy(os.path.join(DataPaths.silver_path, file), os.path.join(DataPaths.gold_path, file))


_ROMAN = 'I II III IV V VI VII VIII IX X'.split()

def run_pipeline(*sections):
    for i, (title, steps) in enumerate(sections):
        print(f"({_ROMAN[i]}) {title}")
        for j, (label, fn) in enumerate(steps):
            print(f"({_ROMAN[i]}.{chr(ord('a') + j)}) {label}...")
            fn()

if __name__ == "__main__":
    ingestion = Ingestion()
    run_pipeline(
        ("Starting ingestion", [
            ("bronze to silver", ingestion.bronze_to_silver),
            ("silver to gold", ingestion.silver_to_gold),
        ]),
    )
