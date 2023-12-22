# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:32:32 2023

@author: Hermes Workstation
"""

import numpy as np
import pandas as pd

# SERIES # 
labels = ['a','b','c']
my_data = [10,20,30]

arr = np.array(my_data)

d = {'a':10,'b':20,'c':30}

pd.Series(data = my_data,index = labels)
pd.Series(my_data,labels)

pd.Series(arr,labels)

d
pd.Series(d)

pd.Series(data=labels)

pd.Series(data=[sum,print,len])

ser1 = pd.Series([1,2,3,4],['USA','Germany','USSR','Japan'])
ser1
ser2 = pd.Series([1,2,5,4],['USA','Japan','Germany','USSR'])
ser2

ser1['Japan']
ser1[0]

ser1 + ser2

# DATA FRAMES #
from numpy.random import randn

np.random.seed(101)
df = pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])
df
df['W']
type(df['W'])
type(df)

df.W
df[['W','X']]

df['New'] = df['W'] + df['X']
df
df.drop('New',axis=1,inplace=True)
df
df.drop('E',axis=0)
df

df.shape
df[['W','X']]
df.loc['A']
df.iloc[2]
df.loc['B','Y']

booldf = df > 0
booldf

df[booldf]
df[df>0]

for i in range(1,4):
    print(i)

df['W'] > 0

df[df['W']>0]['X']
df[(df['W']<0) & (df['Y']>0)] # We can not use 'and' for comparing multiple values
df[(df['W']<0) | (df['Y']>0)] # We can not use 'or' for comparing multiple values

df.reset_index()

newindex = 'CA NY WU CR OR'.split()
newindex
df['States'] = newindex
df
df.set_index('States')

outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)

df = pd.DataFrame(randn(6,2),hier_index,['A','B'])
df
df.loc['G1'].loc[1]
df.index.names = ['Groups','Numbers']
df.xs('G1')
df.xs(1,level='Numbers')

# MISSIN DATA #
d = {'A':[1,2,np.nan],'B':[5,np.nan,np.nan],'C':[1,2,3]}
df = pd.DataFrame(d)
df
df.dropna(thresh=2)
df.fillna(value='Fill value')
df['A'].fillna(value=df['A'].mean())

# GROUPBY #
data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
        'person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
        'Sales':[200,120,340,124,243,350]}
data
df = pd.DataFrame(data)
df
byComp = df.groupby('Company')
byComp
byComp.mean()
byComp.sum()
byComp.std()
byComp.sum().loc['FB']
df.groupby('Company').sum().loc['FB']

df.groupby('Company').count()
df.groupby('Company').max()
df
df.groupby('Company').describe().transpose()['FB']

# MERGIN, JOINING AND CONCATENATING #
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']},
                         index=[4, 5, 6, 7]) 
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                        'B': ['B8', 'B9', 'B10', 'B11'],
                        'C': ['C8', 'C9', 'C10', 'C11'],
                        'D': ['D8', 'D9', 'D10', 'D11']},
                        index=[8, 9, 10, 11])

pd.concat([df1,df2,df3],axis=1)

left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
   
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})

left
right

pd.merge(left,right,how='inner',on='key')

left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3']})
    
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                               'key2': ['K0', 'K0', 'K0', 'K0'],
                                  'C': ['C0', 'C1', 'C2', 'C3'],
                                  'D': ['D0', 'D1', 'D2', 'D3']})

left
pd.merge(left,right,how='right',on=['key1','key2'])

# OPERATIONS #
df = pd.DataFrame({'col1':[1,2,3,4],
                   'col2':[444,555,666,444],
                   'col3':['abc','def','ghi','xyz']})
df.head()

df['col2'].nunique()
df['col2'].value_counts()

df[df['col1']>2]

def times2(x):
    return x*2

df['col1'].apply(times2)
df['col3'].apply(len)
df['col2'].apply(lambda x: x*2)
df
df.drop('col1',axis=1,inplace=True)
df
df.columns
df.index
df
df.sort_values('col2')
df.isnull()

data = {'A':['foo','foo','foo','bar','bar','bar'],
     'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]}

df = pd.DataFrame(data)
df
df.pivot_table('D',['A','B'],['C'])

# DATA INPUT AND OUTPUT #
# CSV, EXCEL, HTML, SQL

pwd
pd.read_csv('example')
df = pd.read_csv('example')
df
df.to_csv('My_output',index=False)
pd.read_csv('My_output')

pd.read_excel('Excel_Sample.xlsx',sheet_name='Sheet1',)
pd.to_xls('Excel_new.xls',sheet_name='NewSheet')

data = pd.read_html('https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/')
type(data)
data[0].head()

from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:')
df.to_sql('my_table',engine)
sqldf = pd.read_sql('my_table',con=engine)
sqldf
