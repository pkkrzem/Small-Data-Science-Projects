# Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Loading and examining the dataset
penguins_df = pd.read_csv("penguins.csv")
print(penguins_df.info())
print(penguins_df.head())

#Checking for missing values
print(penguins_df.isna().sum()/len(penguins_df))

#Dropping missing values
penguins_clean=penguins_df.dropna()

#Checking for outliers - boxplot
penguins_clean.boxplot()
plt.show()

#Defining function for finding outliers
def find_outliers_IQR(dataframe,column_name):

   Q1=dataframe[column_name].quantile(0.25)
   Q3=dataframe[column_name].quantile(0.75)
   IQR=Q3-Q1
   is_lower=dataframe[column_name]<(Q1-(1.5*IQR))
   is_upper=dataframe[column_name]>(Q3+(1.5*IQR))
   outliers= dataframe[column_name].loc[is_lower | is_upper]
   print(column_name, ':', len(outliers))
   return outliers

#Using data without categorical features
penguins_df_cat=penguins_clean.drop('sex', axis=1)

#Finding outliers
for column in list(penguins_df_cat.columns):
   find_outliers_IQR(penguins_clean,column)

#Finding outliers values
to_drop_value=list(find_outliers_IQR(penguins_clean,'flipper_length_mm'))

#Finding outliers indices
idx=[]
for value in to_drop_value:
   idx.append(penguins_clean[penguins_clean['flipper_length_mm'] == value].index.tolist())

#Dropping outliers
idx = [index for sublist in idx for index in sublist]
penguins_clean = penguins_clean.drop(index=idx, axis=0)

#Creating dummy variables for categorical features
penguins_clean=pd.get_dummies(penguins_clean, columns=['sex'])

#Dropping categorical features
penguins_clean=penguins_clean.drop('sex_.', axis=1)

#instanciting Standard Scaler
scaler=StandardScaler()

#scaling data
penguins_preprocessed=scaler.fit_transform(penguins_clean)

#instanciting PCA
pca=PCA()

#finding number of components
pca.fit(penguins_preprocessed)
n_components=sum(pca.explained_variance_ratio_>0.1)

#performing PCA
pca=PCA(n_components=n_components)
penguins_PCA=pca.fit_transform(penguins_preprocessed)

#generating elbow plot
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(penguins_PCA)
                for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$")
plt.ylabel("Inertia")
plt.grid()
plt.show()

#Saving the optimal number of clusters
n_cluster=4

#instanciting kmeans with the optimal number of clusters
kmeans=KMeans(n_clusters=n_cluster, random_state=42).fit(penguins_PCA)

#visualizing clusters using the first two principle components
plt.scatter(x=penguins_PCA[:, 0], y=penguins_PCA[:, 1],c=kmeans.labels_)
plt.show()

#adding label column extracted from the k-means clustering
penguins_clean['label']=kmeans.labels_

#creating a list with the names of the numeric columns 
num_columns=penguins_clean.iloc[:,0:4].columns

#creating a statistical table with penguins' stats
stat_penguins=penguins_clean.groupby('label')[num_columns].mean()






