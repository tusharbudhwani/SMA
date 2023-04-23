import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('sample.csv')
print(df.shape)

print(df.info())

# dropping null values
df = df[df['Tweet Location'].notnull()]
print(df)

# creating a location list
locations_list = []
for location in df['Tweet Location']:
    locations_list.append(location)

# Creating a frequency distribution
location_frequency = pd.Series(locations_list).value_counts()

print(location_frequency.head(10))

# Visualize
location_frequency.head(20).plot(kind='bar')
plt.show()
