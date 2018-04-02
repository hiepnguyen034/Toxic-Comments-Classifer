import pandas as pd
import matplotlib as plt
import wordcloud 
from wordcloud import WordCloud
import seaborn as sns

data=pd.read_csv('toxic comments.csv')
data.head()

def the_word(type):
     words=''
     df=data[data[type]==1]['comment_text']
     for i in df:
         i=i.lower()
         words=words+i+' '
     wordcloud_vis = WordCloud(collocations=False,width=800, height=600).generate(words)
     plt.figure( figsize=(10,8), facecolor='k')
     plt.imshow(wordcloud_vis)
     plt.axis("off")
     plt.tight_layout(pad=0)
     plt.show()
         
the_word('threat')
the_word('toxic')
the_word('severe_toxic')
the_word('obscene')
the_word('identity_hate')

corr=data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,
            xticklabels=list(corr),
            yticklabels=list(corr), annot=True)