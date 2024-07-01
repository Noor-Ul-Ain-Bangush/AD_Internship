#                  <<<<----------ML-WEEk 1.1------------>>>>
#'pandas' lib used for data manipulation & analysis
import pandas as pd

# Use a raw string by prefixing the path with r
df = pd.read_csv(r'E:\A&D_Intership\ML-WEEK 1.1\Iris.csv')
print(df.head())

#To get the basic info of the dataset
df.info()

#To display descriptive statistics
print(df.describe())

#To display the count of each sepal
print(df['Species'].value_counts())


#                                   <-----Visualization------>
#'matplotlib' is used for creating visualizations
import matplotlib.pyplot as plt

# Create histograms
df.hist(figsize=(10, 8))
plt.show()

#Create Scatter plots
plt.figure(figsize=(10,8))
plt.scatter(df['SepalLengthCm'], df['SepalWidthCm'], c='r', label='Sepal_Length vs Sepal_width')
plt.scatter(df['PetalLengthCm'], df['PetalWidthCm'], c='b', label='Petal_Length vs Petal_width')
plt.xlabel('Length')
plt.ylabel('Width')
plt.legend
plt.show()

#Visualize the Dataset with Hue Based on the Species
#'seaborn' library is used for attractive & informative statistical graphs
import seaborn as sns

# Create a pairplot
sns.pairplot(df, hue='Species')
plt.show()


