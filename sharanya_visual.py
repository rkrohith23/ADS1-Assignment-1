import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

file_path = "electronic-file-series-2004-2013.csv"

df = pd.read_csv(file_path)

print(df.head())

df.columns

df.shape

df.info()

df.isna().sum()

df.dropna(inplace = True)

encoded_columns = ['Prefix', 'FileOffice','CurrentOwner', 'UsagePeriod', 'BusinessFunctionCovered']
                   
lab_encoder = LabelEncoder()

lab_encoder = LabelEncoder()
for column in encoded_columns:
    df[column] = lab_encoder.fit_transform(df[column])
    
df.head(10)

#Line plot

#using def function
def line_plot(dataframe):
    plt.figure(figsize=(10, 8))
    plt.plot(dataframe['UsagePeriod'], dataframe['CurrentOwner'], color='red')
    plt.xlabel('UsagePeriod')
    plt.ylabel('CurrentOwner')
    plt.title('Line Plot')
    plt.legend(title='relationship between variables')
    plt.show()

line_plot(df)


def create_line_plot(dataframe, x_col, y_col, title, x_lab, y_lab, legend):
    plt.figure(figsize=(10, 8))
    plt.plot(dataframe[x_col], dataframe[y_col], color='brown')
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    plt.legend(title=legend)
    plt.show()

create_line_plot(df, 'Prefix', 'FileOffice', 'Line Plot', 'Prefix', 'FileOffice', 'relationship between variables')


#Scatter plot

#using def function
def scatter_plot_(data, x_lab, y_lab, title, color='purple'):
    plt.figure(figsize=(10, 8))
    plt.scatter(data[x_lab], data[y_lab], color=color)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    plt.show()

scatter_plot_(df, 'NumberOfDocumentsHeld', 'BusinessFunctionCovered', 'Scatter Plot')

def scatter_plot(x, y, x_lab, y_lab, title):
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, color='blue')
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    plt.show()

x_data = df['CurrentOwner']
y_data = df['NumberOfFilesHeld']
x_lab = 'CurrentOwner'
y_lab = 'NumberOfFilesHeld'
plot = 'Scatter Plot'

scatter_plot(x_data, y_data, x_lab, y_lab, plot)



#Bar plot

#using def function
def plot_bar_chart(data_frame, x_col, y_col, title, color='orange'):
    plt.figure(figsize=(10, 8))
    plt.bar(data_frame[x_col], data_frame[y_col], color=color)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.show()

plot_bar_chart(df, 'Prefix', 'BusinessFunctionCovered', 'Bar Plot')


def bar_plot(data_frame, x_colm, y_colm, x_lab, y_lab, title, figsize=(10, 8), color='green'):
    plt.figure(figsize=figsize)
    plt.bar(data_frame[x_colm], data_frame[y_colm], color=color)
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    plt.show()

bar_plot(df, 'Prefix', 'NumberOfFilesHeld', 'Prefix', 'NumberOfFilesHeld', 'Bar Plot')










