import pandas as pd

df = pd.read_csv("data/final_dataset_2024_stratified.csv")

top5 = df.nlargest(5, "methane_rate")[["folder_id", "methane_rate", "json_lat", "json_lon"]]
print(top5)

