import pandas as pd
import matplotlib.pyplot as plt

# Read in the CSV file
df = pd.read_csv('ghana_nigeria_takedown_tweets.csv')

# Group the data by hashtags and user groups, and count the occurrences
hashtags_by_group = df.groupby(
    ['tweet_client_name', 'hashtags']).size().reset_index(name='count')

# Plot a horizontal bar chart for each user group
for group in df['tweet_client_name'].unique():
    group_data = hashtags_by_group[hashtags_by_group['tweet_client_name'] == group].sort_values('count', ascending=False).head(10)
    ax = group_data.plot(kind='barh', x='hashtags', y='count',
                         legend=False, title=f'Top Hashtags for {group}')
    ax.set_xlabel('Frequency')
    plt.tight_layout()
    plt.show()