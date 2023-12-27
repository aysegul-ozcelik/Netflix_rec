from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import math as math
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import plotly as py
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot, plot
import seaborn as sns
import missingno
from matplotlib.text import Text
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, Polygon
from plotly import graph_objects as go
from collections import Counter
from scipy.stats import norm
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.preprocessing import MultiLabelBinarizer # Similar to One-Hot Encoding
from sklearn.feature_extraction.text import TfidfVectorizer #TfidfVectorizer - Transforms text to feature vectors that can be used as input to estimator.
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import MiniBatchKMeans
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer  # Similar to One-Hot Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df=pd.read_csv('netflix_titles.csv')
df.head()

df.columns

df_1 = df[['title','type','listed_in','description']]

df_1['tags'] = df_1['description'] + df_1['listed_in']

new_data  = df_1.drop(columns=['listed_in','description' ])
new_data

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(max_features=8807, stop_words='english')

vector=cv.fit_transform(new_data['tags'].values.astype('U')).toarray()
vector.shape

from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vector)
similarity


new_data[new_data['title']=="Dick Johnson Is Dead"].index[0]


distance = sorted(list(enumerate(similarity[0])), reverse=True, key=lambda vector:vector[1])
for i in distance[0:5]:
    print(new_data.iloc[i[0]].title)


def recommand(df_1):
    index=new_data[new_data['title']==df_1].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector:vector[1])
    for i in distance[0:5]:
        print(new_data.iloc[i[0]].title)

recommand("Stranger Things")

recommand("Black Mirror")

recommand('Jaws')


import pickle
pickle.dump(new_data, open('movies_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
pickle.load(open('movies_list.pkl', 'rb'))


recommand("The Walking Dead")