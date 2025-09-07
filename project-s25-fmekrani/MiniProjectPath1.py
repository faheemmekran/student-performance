import pandas
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


'''
 The following is the starting code for path1 for data reading to make your first step easier.
 'dataset_1' is the clean data for path1.
'''

with open('behavior-performance.txt','r') as f:
    raw_data = [x.strip().split('\t') for x in f.readlines()]
df = pandas.DataFrame.from_records(raw_data[1:],columns=raw_data[0])
df['VidID']       = pandas.to_numeric(df['VidID'])
df['fracSpent']   = pandas.to_numeric(df['fracSpent'])
df['fracComp']    = pandas.to_numeric(df['fracComp'])
df['fracPlayed']  = pandas.to_numeric(df['fracPlayed'])
df['fracPaused']  = pandas.to_numeric(df['fracPaused'])
df['numPauses']   = pandas.to_numeric(df['numPauses'])
df['avgPBR']      = pandas.to_numeric(df['avgPBR'])
df['stdPBR']      = pandas.to_numeric(df['stdPBR'])
df['numRWs']      = pandas.to_numeric(df['numRWs'])
df['numFFs']      = pandas.to_numeric(df['numFFs'])
df['s']           = pandas.to_numeric(df['s'])
dataset_1 = df
#print(dataset_1[15620:25350].to_string()) #This line will print out the first 35 rows of your data

#question1: How well can the students be naturally grouped or clustered by their video-watching behavior 
user_video_counts = dataset_1.groupby('userID').size()
valid_users = user_video_counts[user_video_counts >= 5].index
filtered_df = dataset_1[dataset_1['userID'].isin(valid_users)].copy()

# Use only the relevant features for clustering
features = ['VidID', 'fracSpent', 'fracComp', 'fracPlayed', 'fracPaused', 'numPauses', 'avgPBR', 'stdPBR', 'numRWs', 'numFFs', 's']

# For clustering, let's exclude the 's' column if you only want behavior
cluster_features = ['fracSpent', 'fracComp', 'fracPlayed', 'fracPaused', 'numPauses', 'avgPBR', 'stdPBR', 'numRWs', 'numFFs']

# KMeans clustering
kmeans6 = KMeans(n_clusters=6, max_iter=300, random_state=50)
filtered_df['cluster6'] = kmeans6.fit_predict(filtered_df[cluster_features])
sil_score6 = silhouette_score(filtered_df[cluster_features], filtered_df['cluster6'])
print("Silhouette Score for 6 clusters:", sil_score6)

# For 7 clusters
kmeans7 = KMeans(n_clusters=7, max_iter=300, random_state=50)
filtered_df['cluster7'] = kmeans7.fit_predict(filtered_df[cluster_features])
sil_score7 = silhouette_score(filtered_df[cluster_features], filtered_df['cluster7'])
print("Silhouette Score for 7 clusters:", sil_score7)

# For 8 clusters
kmeans8 = KMeans(n_clusters=8, max_iter=300, random_state=50)
filtered_df['cluster8'] = kmeans8.fit_predict(filtered_df[cluster_features])
sil_score8 = silhouette_score(filtered_df[cluster_features], filtered_df['cluster8'])
print("Silhouette Score for 8 clusters:", sil_score8)

# Elbow method
wcss = []
for k in range(1, 11):
    kmeans_tmp = KMeans(n_clusters=k, random_state=50)
    kmeans_tmp.fit(filtered_df[cluster_features])
    wcss.append(kmeans_tmp.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


#quesiton 2: 
user_agg = filtered_df.groupby('userID').agg({
    'fracSpent': 'mean',
    'fracPaused': 'mean',
    's': ['sum', 'mean']
}).reset_index()
user_agg.columns = ['userID', 'avg_fracSpent', 'avg_fracPause', 'sum_score', 'final_score']

# Model 1: Linear regression for avg_fracSpent vs final_score
model1 = LinearRegression()
model1.fit(user_agg[['avg_fracSpent']], user_agg['final_score'])
yp1 = model1.predict(user_agg[['avg_fracSpent']])
r2_1 = r2_score(user_agg['final_score'], yp1)
mse_1 = mean_squared_error(user_agg['final_score'], yp1)
print("Model 1 (fracSpent): R^2 =", r2_1, "; MSE =", mse_1)

# Model 2: Linear regression for avg_fracPause vs final_score
model2 = LinearRegression()
model2.fit(user_agg[['avg_fracPause']], user_agg['final_score'])
yp2 = model2.predict(user_agg[['avg_fracPause']])
r2_2 = r2_score(user_agg['final_score'], yp2)
mse_2 = mean_squared_error(user_agg['final_score'], yp2)
print("Model 2 (fracPaused): R^2 =", r2_2, "; MSE =", mse_2)

#question 3: visualization
plt.figure()
sns.scatterplot(x='avg_fracSpent', y='final_score', data=user_agg)
plt.title('Average fracSpent vs Final Score')
plt.show()

plt.figure()
sns.scatterplot(x='avg_fracPause', y='final_score', data=user_agg)
plt.title('Average fracPaused vs Final Score')
plt.show()

plt.figure()
sns.scatterplot(x='fracSpent', y='fracPaused', data=filtered_df)
plt.title('fracSpent vs fracPaused (All Observations)')
plt.show()