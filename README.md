# Learn_Python_Tutorial

In this basic .py notebook, I have uploaded a csv and have performed very initial analysis, the purpose of this notebook is to understand basic coding techniques.
# 1. Import all libraries:

import pandas as pd
import pandas
import math
import statistics as stats
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 2. This code needs to be changed according your own drive and path:
You can find the csv in the repository, save it in your google drive and rephrase this code to suit your path file/directory,

from google.colab import drive
drive.mount("/content/drive")

#Load CSV file into a DataFrame
file_path = '/content/drive/MyDrive/Colab Notebooks/007 car-sales.csv'
cdata = pd.read_csv(file_path)

# 3. This will show first few rows of the data:
To Display the first few rows of the DataFrame

print(cdata.head())

# 4. To begin my analysis i will strat with basics of getting to know my data
1. Browse data by simply writing the name of data as saved in my case it will be cdata


cdata

2.To find out about the length of my variables

cdata.index

cdata.shape

3. Basics about vars

cdata.describe()

4. To know the data type of variables(wether a var is string , bool, numeric, etc.)

cdata.dtypes

5.To know if there are any missing values in the data

cdata.info()

cdata.isnull().sum()
# Replacing Null values:
Note: there are no null values in this data, however if we had encountered any nulls we could have performed a) forward fill or b) backwards fill, c)replacing with mean , mode or, median

a) Forward Fill
f_cdata = cdata.fillna(method = "ffill", inpace = True)
b) Backward Fill
b_cdata = cdata.fillna(method = "bfill", inplace = True)
c) If i want to use mean values to fill missing values
mean_cdata =  cdata.fillna(data["bill_depth_mm"].mean(), inplace = True)
data["bill_depth_mm"].mean()

#Its not always important to replace the missing values,  i can simply drop it too prevent skewness and miss representation.
data.dropna()

6. To quickly get mean of data I use this command, *Note: This only works for numeric values in the data

cdata.mean()

7. cdata.sum(), only works if data type is numeric , incase of strings it just sum the strin values.
cdata.sum()

#these code can be used to perform sum on individual columns
cdata["Doors"].sum()

cdata["Odometer (KM)"].sum()

#this code will replace charachters from price column and chnage its type from string to float
cdata['Price'] = cdata['Price'].str.replace('[\$,]', '', regex=True).astype(float)

#browsing data will now show that it has changed to float
cdata
#again we can use sun command now
cdata["Price"].sum()


8.To convert other strings to numeric i am coding them, this command will directly code them

from sklearn.preprocessing import LabelEncoder

#Initialize LabelEncoders for 'Make' and 'Colour'
make_encoder = LabelEncoder()
colour_encoder = LabelEncoder()

#Encode 'Make' and 'Colour' columns and store them in new columns
cdata['Make_Label'] = make_encoder.fit_transform(cdata['Make']) + 1
cdata['Colour_Label'] = colour_encoder.fit_transform(cdata['Colour']) + 1
#I am adding plus one in this code so that the encoding dosent start from zero but 1.
cdata

#I am encoding string values to run statictical analysis and making them numeric by coding them,

#I will run mode command for encoded values, as mean will not be a good meansure for measuring central tendencies
cdata['Colour_Label'].mode()

cdata['Make_Label'].mode()

#Interpretation for result: Mode here means that these are most occuring values, for color mean White or 5 coloured cars are most, for make 4 means Toyota is the car having most frequeny

#  Group by and Crosstabulation:

groupby_cdata = cdata.groupby(["Make", "Colour"])
groupby_cdata

groupby_cdata["Price","Odometer (KM)", "Doors"].agg([np.mean, "count"])

## Interpretation of the groupby data columns
### Make and Colour Combinations:

The data is grouped by two categorical columns, "Make" and "Colour," creating unique combinations. For example, there's one combination with "BMW" and "Black," one with "Honda" and "Blue," and so on.

### Aggregated Statistics:

Price (mean): This column represents the mean (average) price for the specific combination of "Make" and "Colour." For instance, for "BMW" cars that are "Black," the average price is $22,000.0.
Price (count): This column shows the count of records (cars) that fall into the specific "Make" and "Colour" category. For example, there is one car that is both "BMW" and "Black."

pd.crosstab(cdata["Make"], cdata["Colour"])

### Cross tabulation:
shows us in which category of Make we have what colour of cars

cdata[["Odometer (KM)", "Doors", "Price"]].corr()

cdata.corr()

# Visulaization for correlation
Create a correlation matrix
corr_matrix = cdata.corr()

Set the figure size
plt.figure(figsize=(8, 6))

Create a heatmap with customized style
sns.set(font_scale=1)
sns.set_style("whitegrid")
sns.color_palette("coolwarm")
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)

Set the title
plt.title("Correlation Heatmap")

Show the plot
plt.show()
# qcut and bins 
cdata["Prcie_bins"] = pd.qcut(cdata["Price"], q = 3)
cdata["Prcie_bins"].value_counts()
#this can be use ful for making bins if needed in future analysis

cdata
