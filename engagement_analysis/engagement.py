import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('MrBeast.csv')

print(df.info())

likes_df = df.sort_values(by='likeCount', ascending=False).head(10)
print(likes_df['likeCount'])

plt.bar(likes_df['content'], likes_df['likeCount'])
plt.xticks(rotation = 90)
plt.show()