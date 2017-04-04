import pandas as pd
import numpy as np

df=pd.read_csv('training.csv')
df.Image=df['Image'].apply(lambda im : np.fromstring(im,sep=' '))
df=df.dropna()
X=np.vstack(df.Image.values)/255
X=X.astype(np.float32)
y=df[df.columns[:-1]].values
y=(y-48)/48 # Normalize the values between [0,1]
y=y.astype(np.float32)
