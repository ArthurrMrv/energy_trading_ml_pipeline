import os 

from enum import Enum

# class DataTypes:
#     INVENTORY_OF_GENERATION 
#     MONTHLY_HOURLY_LOAD_VALUES
#     PHYSICAL_ENERGY_POWER_FLOWS

class DataTemplates:
    INVENTORY_OF_GENERATION = "inventory_of_generation"
    MONTHLY_HOURLY_LOAD_VALUES = "monthly_hourly_load_values"
    PHYSICAL_ENERGY_POWER_FLOWS = "physical_energy_power_flows"
    EUROPEAN_WHOLESALE_ELECTRICITY_PRICE_DATA_DAILY = "european_wholesale_electricity_price_data_daily"

class DataPaths:
    data_path = "backend/data"
    bronze_path = os.path.join(data_path, "bronze")
    silver_path = os.path.join(data_path, "silver")
    gold_path = os.path.join(data_path, "gold")

    def is_ofType(self, file: str):
        if file.startswith("inventory_of_generation"):
            return "inventory_of_generation"
        elif file.startswith("monthly_hourly_load_values"):
            return "monthly_hourly_load_values"
        elif file.startswith("physical_energy_power_flows"):
            return "physical_energy_power_flows"
        else:
            return None