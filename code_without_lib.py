import pandas as pd
import numpy as np
df=pd.read_csv('ML-Set1.csv')
path='ML-Set2.csv'
newvar=df.head()
data=pd.DataFrame(newvar)
numeric_columns=df.select_dtypes(include=[np.number])
string_columns = df.select_dtypes(include=[np.object_])
total={}
count={}
for index , values in numeric_columns.iterrows():
    for  column , value in values.items():
        if pd.notna(value):
            total[column] = total.get(column,0)+value
            count[column] = count.get(column,0)+1
mean={column: total[column]/count[column] for column in total}
for i , value in mean.items():
    numeric_columns.fillna(value , inplace=True)
numeric_columns.to_csv(path,index=False)
new_df=pd.read_csv(path)
country_column=string_columns['Country']
mode=country_column.mode().iloc[0]
country_column.fillna(mode,inplace=True)
new_df['Country'] = country_column
new_df.to_csv(path,index=True)
unique_country = new_df['Country'].unique()
for country in unique_country:
    new_df[country] =(new_df['Country'] == country).astype(int)

for each in new_df:
    new_df['Quality'] = (string_columns['Quality'] == "good").astype(int)
new_df.to_csv(path,index=True)  


        



