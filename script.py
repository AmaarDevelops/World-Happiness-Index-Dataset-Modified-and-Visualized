import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


# Get the folder where the script is running
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, '2019.csv')
df = pd.read_csv(file_path)


df_head = df.head()
df_info = df.info()
df_describe = df.describe()

null_values = df.isnull().sum()


renamed_columns = df.rename(columns={'Country or region' : 'country',
                                     'GDP per capita' : 'gdp',
                                     "Freedom to make life choices" : "freedom",
                                     "Perceptions of corruption" : "corruption"},inplace=True)

numerical_explanatory_columns = df.select_dtypes(include=['float64','int64']).columns.drop(['Overall rank','Score'])

df['Sum_of_factors'] = df[numerical_explanatory_columns].sum(axis=1)

def categorize_happiness(score):
    if score > 7:
        return 'High'
    elif 5 <= score <=7:
        return 'Medium'
    else:
        return 'Low'


df['Happiness_Level'] = df['Score'].apply(categorize_happiness)


top_10_happiest_countries = df.sort_values(by='Overall rank',ascending=True).head(10)
top_10_happiest_countries_df = top_10_happiest_countries.reset_index()

top_10_countries_with_least_corruption = df.sort_values(by='corruption',ascending=True).head(10)['country']


#Print Summaries
print("Df head:-",df_head)
print("Df Information:-",df_info)
print("Df description:-",df_describe)
print("Null Values:-",null_values)
print("Renamed Columns:-",renamed_columns)
print("\n Top 10 happiest countries:-",top_10_happiest_countries)
print("Top 10 countries with least corruptions:-",top_10_countries_with_least_corruption)

df.to_csv('Modified-Visualized-World happiness index-dataset.csv',index=False)

visuals = 'visuals'
if not os.path.exists(visuals):
    os.makedirs(visuals)

#Visualization
plt.figure()
sns.barplot(x=top_10_happiest_countries_df['Score'],y=top_10_happiest_countries_df['country'],palette='deep')
plt.title("Top 10 happiest countries scores")
plt.savefig(os.path.join(visuals,'Top 10 happiest countries with scores.png'))


plt.figure()
sns.histplot(x='Score',data=df,palette='dark')
plt.title("Happiness score distrbution")
plt.savefig(os.path.join(visuals,'Happiness score distribution.png'))

plt.figure()
sns.heatmap(df[numerical_explanatory_columns].corr(),annot=True)
plt.title("Numerical Columns correlation")
plt.savefig(os.path.join(visuals,"Numericals columns correlation.png"))

plt.figure()
sns.scatterplot(x='gdp',y='Score',data=df,palette='muted')
plt.title("GDP vs Happiness Score")

plt.figure()
sns.scatterplot(x='freedom',y='Score',hue='country',data=df)
plt.title("Freedom Vs Score")


plt.figure()
sns.barplot(x=top_10_countries_with_least_corruption.index,y=top_10_countries_with_least_corruption.values,palette='deep')
plt.title("Top 10 countries with least corruption")
plt.savefig(os.path.join(visuals,"Top 10 countries with least corruption.png"))

plt.figure()
sns.countplot(x='Happiness_Level',data=df,palette='pastel')
plt.title("Happiness level dsitribution (High/Medium/Low)")
plt.savefig(os.path.join(visuals,"Happiness level distribution.png"))

plt.show()