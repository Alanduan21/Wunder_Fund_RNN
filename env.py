import pandas as pd
import numpy as np



df = pd.read_parquet("competition_package/datasets/train.parquet")

print(df.head())