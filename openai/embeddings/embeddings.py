# OpenAI’s text embeddings measure the relatedness of text strings. Embeddings are commonly used for:

# Search (where results are ranked by relevance to a query string)
# Clustering (where text strings are grouped by similarity)
# Recommendations (where items with related text strings are recommended)
# Anomaly detection (where outliers with little relatedness are identified)
# Diversity measurement (where similarity distributions are analyzed)
# Classification (where text strings are classified by their most similar label)

# An embedding is a vector (list) of floating point numbers. The distance between two vectors measures their relatedness.
# Small distances suggest high relatedness and large distances suggest low relatedness.

from typing import List
from openai import OpenAI
import pandas as pd
import numpy as np
from torch import cosine_similarity


client = OpenAI()

response = client.chat.completions.create(
    input = "Your text string goes here",
    model = "text-embedding-3-small"
)


#### Get_embeddings_from_dataset

def get_embedding(text, model = "text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model = model).data[0].embedding

df  = pd.read_csv('output/embedded_1k_reviews.csv')

df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-3-small'))
df.to_csv('output/embedded_1k_reviews.csv', index=False)

df['ada_embedding'] = df.ada_embedding.apply(eval).apply(np.array)


################ USE CASES ################


######## Reducing embedding dimensions

def normalize_12(x):
    x = np.array(x)

    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x,2,axis = 1, keepdims = True)
        return np.where(norm == 0, x, x / norm)
    
response2 = client.chat.completions.create(
    model = "text-embedding-3-small",
    input = "Testing 123",
    encoding_format = "float"
)

cut_dim = response2.data[0].embedding[:256]
norm_dim = normalize_12(cut_dim)

######## Question Answering using embedding-based search

wikipedia_article_on_curling = """Curling at the 2022 Winter Olympics

Article
Talk
Read
Edit
View history
From Wikipedia, the free encyclopedia
Curling
at the XXIV Olympic Winter Games
Curling pictogram.svg
Curling pictogram
Venue	Beijing National Aquatics Centre
Dates	2–20 February 2022
No. of events	3 (1 men, 1 women, 1 mixed)
Competitors	114 from 14 nations
← 20182026 →
Men's curling
at the XXIV Olympic Winter Games
Medalists
1st place, gold medalist(s)		 Sweden
2nd place, silver medalist(s)		 Great Britain
3rd place, bronze medalist(s)		 Canada
Women's curling
at the XXIV Olympic Winter Games
Medalists
1st place, gold medalist(s)		 Great Britain
2nd place, silver medalist(s)		 Japan
3rd place, bronze medalist(s)		 Sweden
Mixed doubles's curling
at the XXIV Olympic Winter Games
Medalists
1st place, gold medalist(s)		 Italy
2nd place, silver medalist(s)		 Norway
3rd place, bronze medalist(s)		 Sweden
Curling at the
2022 Winter Olympics
Curling pictogram.svg
Qualification
Statistics
Tournament
Men
Women
Mixed doubles
vte
The curling competitions of the 2022 Winter Olympics were held at the Beijing National Aquatics Centre, one of the Olympic Green venues. Curling competitions were scheduled for every day of the games, from February 2 to February 20.[1] This was the eighth time that curling was part of the Olympic program.

In each of the men's, women's, and mixed doubles competitions, 10 nations competed. The mixed doubles competition was expanded for its second appearance in the Olympics.[2] A total of 120 quota spots (60 per sex) were distributed to the sport of curling, an increase of four from the 2018 Winter Olympics.[3] A total of 3 events were contested, one for men, one for women, and one mixed.[4]

Qualification
Main article: Curling at the 2022 Winter Olympics – Qualification
Qualification to the Men's and Women's curling tournaments at the Winter Olympics was determined through two methods (in addition to the host nation). Nations qualified teams by placing in the top six at the 2021 World Curling Championships. Teams could also qualify through Olympic qualification events which were held in 2021. Six nations qualified via World Championship qualification placement, while three nations qualified through qualification events. In men's and women's play, a host will be selected for the Olympic Qualification Event (OQE). They would be joined by the teams which competed at the 2021 World Championships but did not qualify for the Olympics, and two qualifiers from the Pre-Olympic Qualification Event (Pre-OQE). The Pre-OQE was open to all member associations.[5]

For the mixed doubles competition in 2022, the tournament field was expanded from eight competitor nations to ten.[2] The top seven ranked teams at the 2021 World Mixed Doubles Curling Championship qualified, along with two teams from the Olympic Qualification Event (OQE) – Mixed Doubles. This OQE was open to a nominated host and the fifteen nations with the highest qualification points not already qualified to the Olympics. As the host nation, China qualified teams automatically, thus making a total of ten teams per event in the curling tournaments.[6]

Summary
Nations	Men	Women	Mixed doubles	Athletes
 Australia			Yes	2
 Canada	Yes	Yes	Yes	12
 China	Yes	Yes	Yes	12
 Czech Republic			Yes	2
 Denmark	Yes	Yes		10
 Great Britain	Yes	Yes	Yes	10
 Italy	Yes		Yes	6
 Japan		Yes		5
 Norway	Yes		Yes	6
 ROC	Yes	Yes		10
 South Korea		Yes		5
 Sweden	Yes	Yes	Yes	11
 Switzerland	Yes	Yes	Yes	12
 United States	Yes	Yes	Yes	11
Total: 14 NOCs	10	10	10	114
Competition schedule

The Beijing National Aquatics Centre served as the venue of the curling competitions.
Curling competitions started two days before the Opening Ceremony and finished on the last day of the games, meaning the sport was the only one to have had a competition every day of the games. The following was the competition schedule for the curling competitions:

RR	Round robin	SF	Semifinals	B	3rd place play-off	F	Final
Date
Event
Wed 2	Thu 3	Fri 4	Sat 5	Sun 6	Mon 7	Tue 8	Wed 9	Thu 10	Fri 11	Sat 12	Sun 13	Mon 14	Tue 15	Wed 16	Thu 17	Fri 18	Sat 19	Sun 20
Men's tournament								RR	RR	RR	RR	RR	RR	RR	RR	RR	SF	B	F	
Women's tournament									RR	RR	RR	RR	RR	RR	RR	RR	SF	B	F
Mixed doubles	RR	RR	RR	RR	RR	RR	SF	B	F												
Medal summary
Medal table
Rank	Nation	Gold	Silver	Bronze	Total
1	 Great Britain	1	1	0	2
2	 Sweden	1	0	2	3
3	 Italy	1	0	0	1
4	 Japan	0	1	0	1
 Norway	0	1	0	1
6	 Canada	0	0	1	1
Totals (6 entries)	3	3	3	9
Medalists
Event	Gold	Silver	Bronze
Men
details	 Sweden
Niklas Edin
Oskar Eriksson
Rasmus Wranå
Christoffer Sundgren
Daniel Magnusson	 Great Britain
Bruce Mouat
Grant Hardie
Bobby Lammie
Hammy McMillan Jr.
Ross Whyte	 Canada
Brad Gushue
Mark Nichols
Brett Gallant
Geoff Walker
Marc Kennedy
Women
details	 Great Britain
Eve Muirhead
Vicky Wright
Jennifer Dodds
Hailey Duff
Mili Smith	 Japan
Satsuki Fujisawa
Chinami Yoshida
Yumi Suzuki
Yurika Yoshida
Kotomi Ishizaki	 Sweden
Anna Hasselborg
Sara McManus
Agnes Knochenhauer
Sofia Mabergs
Johanna Heldin
Mixed doubles
details	 Italy
Stefania Constantini
Amos Mosaner	 Norway
Kristin Skaslien
Magnus Nedregotten	 Sweden
Almida de Val
Oskar Eriksson
Teams
Men
 Canada	 China	 Denmark	 Great Britain	 Italy
Skip: Brad Gushue
Third: Mark Nichols
Second: Brett Gallant
Lead: Geoff Walker
Alternate: Marc Kennedy

Skip: Ma Xiuyue
Third: Zou Qiang
Second: Wang Zhiyu
Lead: Xu Jingtao
Alternate: Jiang Dongxu

Skip: Mikkel Krause
Third: Mads Nørgård
Second: Henrik Holtermann
Lead: Kasper Wiksten
Alternate: Tobias Thune

Skip: Bruce Mouat
Third: Grant Hardie
Second: Bobby Lammie
Lead: Hammy McMillan Jr.
Alternate: Ross Whyte

Skip: Joël Retornaz
Third: Amos Mosaner
Second: Sebastiano Arman
Lead: Simone Gonin
Alternate: Mattia Giovanella

 Norway	 ROC	 Sweden	 Switzerland	 United States
Skip: Steffen Walstad
Third: Torger Nergård
Second: Markus Høiberg
Lead: Magnus Vågberg
Alternate: Magnus Nedregotten

Skip: Sergey Glukhov
Third: Evgeny Klimov
Second: Dmitry Mironov
Lead: Anton Kalalb
Alternate: Daniil Goriachev

Skip: Niklas Edin
Third: Oskar Eriksson
Second: Rasmus Wranå
Lead: Christoffer Sundgren
Alternate: Daniel Magnusson

Fourth: Benoît Schwarz
Third: Sven Michel
Skip: Peter de Cruz
Lead: Valentin Tanner
Alternate: Pablo Lachat

Skip: John Shuster
Third: Chris Plys
Second: Matt Hamilton
Lead: John Landsteiner
Alternate: Colin Hufman

Women
 Canada	 China	 Denmark	 Great Britain	 Japan
Skip: Jennifer Jones
Third: Kaitlyn Lawes
Second: Jocelyn Peterman
Lead: Dawn McEwen
Alternate: Lisa Weagle

Skip: Han Yu
Third: Wang Rui
Second: Dong Ziqi
Lead: Zhang Lijun
Alternate: Jiang Xindi

Skip: Madeleine Dupont
Third: Mathilde Halse
Second: Denise Dupont
Lead: My Larsen
Alternate: Jasmin Lander

Skip: Eve Muirhead
Third: Vicky Wright
Second: Jennifer Dodds
Lead: Hailey Duff
Alternate: Mili Smith

Skip: Satsuki Fujisawa
Third: Chinami Yoshida
Second: Yumi Suzuki
Lead: Yurika Yoshida
Alternate: Kotomi Ishizaki

 ROC	 South Korea	 Sweden	 Switzerland	 United States
Skip: Alina Kovaleva
Third: Yulia Portunova
Second: Galina Arsenkina
Lead: Ekaterina Kuzmina
Alternate: Maria Komarova

Skip: Kim Eun-jung
Third: Kim Kyeong-ae
Second: Kim Cho-hi
Lead: Kim Seon-yeong
Alternate: Kim Yeong-mi

Skip: Anna Hasselborg
Third: Sara McManus
Second: Agnes Knochenhauer
Lead: Sofia Mabergs
Alternate: Johanna Heldin

Fourth: Alina Pätz
Skip: Silvana Tirinzoni
Second: Esther Neuenschwander
Lead: Melanie Barbezat
Alternate: Carole Howald

Skip: Tabitha Peterson
Third: Nina Roth
Second: Becca Hamilton
Lead: Tara Peterson
Alternate: Aileen Geving

Mixed doubles
 Australia	 Canada	 China	 Czech Republic	 Great Britain
Female: Tahli Gill
Male: Dean Hewitt

Female: Rachel Homan
Male: John Morris

Female: Fan Suyuan
Male: Ling Zhi

Female: Zuzana Paulová
Male: Tomáš Paul

Female: Jennifer Dodds
Male: Bruce Mouat

 Italy	 Norway	 Sweden	 Switzerland	 United States
Female: Stefania Constantini
Male: Amos Mosaner

Female: Kristin Skaslien
Male: Magnus Nedregotten

Female: Almida de Val
Male: Oskar Eriksson

Female: Jenny Perret
Male: Martin Rios

Female: Vicky Persinger
Male: Chris Plys
"""

query = f"""Use the below article on the 2022 Winter Olympics to answer the subsequent question. If the answer cannot be found, write "I don't know."

Article:
\"\"\"
{wikipedia_article_on_curling}
\"\"\"

Question: Which athletes won the gold medal in curling at the 2022 Winter Olympics?"""

response3 = client.chat.completions.create(
    messages = [
        {"role" : "system","content" : "You answer questions about the 2022 Winter Olympics."},
        {"role" : "user","content" : "query"}
    ],
    model = "gpt-3.5-turbo",
    temperature = 0
)

print(response3.choices[0].message.content)

######## Text Search using embeddings

def search_reviews(df,product_description, n=3,pprint=True):
    embedding = get_embedding(product_description, model='text-embedding-3-small')
    df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values('similarities', ascending=False).head(n)
    return res

res = search_reviews(df, 'delicious beans', n=3)

######## Code search using embeddings

df['code_embedding'] = df['code'].apply(lambda x: get_embedding(x, model='text-embedding-3-small'))

def search_functions(df, code_query, n=3, pprint=True, n_lines=7):
   embedding = get_embedding(code_query, model='text-embedding-3-small')
   df['similarities'] = df.code_embedding.apply(lambda x: cosine_similarity(x, embedding))

   res = df.sort_values('similarities', ascending=False).head(n)
   return res
res = search_functions(df, 'Completions API tests', n=3)

######## Recommendations using embeddings 

def embedding_from_string(
    string: str,
    model: str = EMBEDDING_MODEL,
    embedding_cache=embedding_cache
) -> list:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]

def recommendations_from_strings(
        strings : List[str],
        index_of_source_string : int,
        model = "text-embedding-3*-small",
) -> List[int]:
    """Return nearest neighbors of a given string."""

    # get the embedding of the source string
    embeddings = [embedding_from_string(string,model = model) for string in strings]

    # get the embedding of the source string
    query_embedding = embeddings[index_of_source_string]

    # get distances between the source embedding and other embeddings
    distances = distances_from_embeddings(query_embedding,embeddings,distance_metric = "cosine")

    # get indices of nearest neighbors (function from embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
    return indices_of_nearest_neighbors


######## Data Visualization in 2D

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    list(df.ada_embedding.values),
    df.Score,
    test_size = 0.2,
    random_state=42
)

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=100)
rfr.fit(X_train, y_train)
preds = rfr.predict(X_test)


######## Classification using the embedding features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)


######## Zero-shot classification

from openai.embeddings_utils import cosine_similarity, get_embedding

df= df[df.Score!=3]
df['sentiment'] = df.Score.replace({1:'negative', 2:'negative', 4:'positive', 5:'positive'})

labels = ['negative', 'positive']
label_embeddings = [get_embedding(label, model=model) for label in labels]

def label_score(review_embedding, label_embeddings):
   return cosine_similarity(review_embedding, label_embeddings[1]) - cosine_similarity(review_embedding, label_embeddings[0])

prediction = 'positive' if label_score('Sample Review', label_embeddings) > 0 else 'negative'


######## Obtaining user and product embeddings for cold-start recommendation

user_embeddings = df.groupby('UserId').ada_embedding.apply(np.mean)
prod_embeddings = df.groupby('ProductId').ada_embedding.apply(np.mean)

######## Clustering

import numpy as np
from sklearn.cluster import KMeans

matrix = np.vstack(df.ada_embedding.values)
n_clusters = 4

kmeans = KMeans(n_clusters = n_clusters, init='k-means++', random_state=42)
kmeans.fit(matrix)
df['Cluster'] = kmeans.labels_


##### How can I tell how many tokens a string has before I embed it?

import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

num_tokens_from_string("tiktoken is great!", "cl100k_base")