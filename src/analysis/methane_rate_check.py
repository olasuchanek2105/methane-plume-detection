import pandas as pd

df = pd.read_csv("data/final_dataset_2024_stratified.csv")

top5 = df.nlargest(5, "methane_rate")[["folder_id", "methane_rate", "json_lat", "json_lon"]]
print(top5)


#                             folder_id  methane_rate  json_lat  json_lon
# 0    00ba2312cab12f013293d0c63f194223           559     45.47     55.09
# 327  bcd498d38cea7ccb05e3ec7727333221           265     46.22     54.53
# 273  9ddd110ce4adbb52fdf323bf22b80993           209     39.42     53.64
# 96   3600207c4ecd24a4f9d35c7e91cbb2dd           203     52.43     43.94
# 292  a9327ae740df6ff3dfdd3983fd80eb79           195    -46.47    -68.99