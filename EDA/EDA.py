import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read the CSV file
df = pd.read_csv('_animes.csv')

# display the dataset head
print(df.head(5), '\n')

# check info of the dataset
print(df.info(), '\n')

# summary
print(df.describe(), '\n')

# Percentage of missing values
print('Percentage of missing values:\n', df.isnull().sum() / df.shape[0] * 100)

# dropping uid as it is of no use
df = df.drop(['uid'], axis=1)

# Visualizing

new_df = df.head(20)

# countplot
sns.countplot(data=new_df, x= 'episodes')
plt.show()

# scatterplot
plt.scatter(x=new_df['episodes'], y=new_df['popularity'])
plt.xlabel('Score')
plt.ylabel('Popularity')
plt.title('Score vs Popularity')
plt.show()

#bar
plt.bar(new_df['title'], new_df['popularity'])
plt.xlabel('Title')
plt.ylabel('Popularity')
plt.title('Popularity of each show')
plt.show()