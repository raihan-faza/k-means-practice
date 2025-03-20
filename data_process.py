import pandas as pd

df = pd.read_csv("penguins.csv")
print(df.info())
print(df.describe())
print(df.isnull().sum())
