import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('trend_analysis.csv')
print(df.info())

# converting date to a datetime column using pd
df['date'] = pd.to_datetime(df['date'])

df.set_index('date', inplace=True)

daily_counts = df.resample('D').count()

plt.plot(daily_counts.index, daily_counts['id'])
plt.xticks(rotation = 90)
plt.show()
