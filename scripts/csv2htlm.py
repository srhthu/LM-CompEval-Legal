"""
Convert the csv table to html table
"""
# %%
import pandas as pd

df = pd.read_csv('../resources/paper_version_f1.csv')
# %%
df2 = df.reset_index()
df2['index'] = df2['index'] + 1
df2 = df2.rename({'index': 'rank'}, axis= 1)
# %%
print(df2.to_html(index = False, float_format=lambda k: f'{k*100:.2f}',
            justify = 'justify'))
# %%
