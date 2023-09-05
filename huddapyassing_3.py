# -*- coding: utf-8 -*-
"""huddapyassing_3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nzSxPM-SwmNIeBKFDwR-j6I4OEvXFReZ
"""

import pandas as pd
import pandas
import math
import statistics as stats
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount("/content/drive")

# Load CSV file into a DataFrame
file_path = '/content/drive/MyDrive/Colab Notebooks/007 car-sales.csv'
cdata = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(cdata.head())

"""#To begin my analysis i will strat with basic statistics
1. Browse data by simply writing the name of data as saved in my case it will be cdata
"""

cdata

"""2.To find out about the length of my variables"""

cdata.index

cdata.shape

"""3. Basics about vars"""

cdata.describe()

"""4. To know the data type of variables(wether a var is string , bool, numeric, etc.)"""

cdata.dtypes

"""5.To know if there are any missing values in the data

"""

cdata.info()

cdata.isnull().sum()

#there were no null values in the data, however if we has encountered any nuls we could have performed forward fill or backwards fill,, replacing with mean , mode or, median

#b_cdata = cdata.fillna(method = "bfill", inplace = True)
#f_cdata = cdata.fillna(method = "ffill", inpace = True)

#If i want to use mean values to fill missing values
#mean_cdata =  cdata.fillna(data["bill_depth_mm"].mean(), inplace = True)
#data["bill_depth_mm"].mean()




#if i have alot of data of million and 10 percent is missing, i can simply drop it to prevent skewness and miss representation.
#data.dropna()

"""6. To quickly get mean of data I use this command, *Note: This only works for numeric values in the data

"""

cdata.mean()

"""7. cdata.sum(), only works if data type is numeric , incase of strings it just sum the strin values.

"""

cdata.sum()

cdata["Doors"].sum()

cdata["Odometer (KM)"].sum()

cdata['Price'] = cdata['Price'].str.replace('[\$,]', '', regex=True).astype(float)

cdata

cdata["Price"].sum()

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoders for 'Make' and 'Colour'
make_encoder = LabelEncoder()
colour_encoder = LabelEncoder()

# Encode 'Make' and 'Colour' columns and store them in new columns
cdata['Make_Label'] = make_encoder.fit_transform(cdata['Make']) + 1
cdata['Colour_Label'] = colour_encoder.fit_transform(cdata['Colour']) + 1
#i am adding plus one in this code so that the encoding dosent start from zero but 1.
cdata

# I am encoding string values to run statictical analysis and making them numeric by coding them,

#I will run mode command for encoded values
cdata['Colour_Label'].mode()

cdata['Make_Label'].mode()

"""#Mode here means that these are most occuring values, for color mean White or 5 coloured cars are most, for make 4 means Toyota is the car having most frequency"""

groupby_cdata = cdata.groupby(["Make", "Colour"])
groupby_cdata

groupby_cdata["Price","Odometer (KM)", "Doors"].agg([np.mean, "count"])

"""#Interpretation of the groupby data columns
#Make and Colour Combinations:

The data is grouped by two categorical columns, "Make" and "Colour," creating unique combinations. For example, there's one combination with "BMW" and "Black," one with "Honda" and "Blue," and so on.

Aggregated Statistics:

Price (mean): This column represents the mean (average) price for the specific combination of "Make" and "Colour." For instance, for "BMW" cars that are "Black," the average price is $22,000.0.
Price (count): This column shows the count of records (cars) that fall into the specific "Make" and "Colour" category. For example, there is one car that is both "BMW" and "Black."
"""

pd.crosstab(cdata["Make"], cdata["Colour"])

"""#Cross tabulation:
shows us in which category of Make we have what colour of cars
"""

cdata[["Odometer (KM)", "Doors", "Price"]].corr()

cdata.corr()

#Visulaization for correlation
# Create a correlation matrix
corr_matrix = cdata.corr()

# Set the figure size
plt.figure(figsize=(8, 6))

# Create a heatmap with customized style
sns.set(font_scale=1)
sns.set_style("whitegrid")
sns.color_palette("coolwarm")
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)

# Set the title
plt.title("Correlation Heatmap")

# Show the plot
plt.show()

cdata["Prcie_bins"] = pd.qcut(cdata["Price"], q = 3)
cdata["Prcie_bins"].value_counts()

cdata