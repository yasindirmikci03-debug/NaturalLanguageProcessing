from utils import authenticate

credentials, PROJECT_ID = authenticate()

REGION = 'us-central1'

import vertexai

vertexai.init(project= PROJECT_ID,location = REGION,credentials=credentials)

#### Embeddings capture meaning

in_1 = "Missing flamingo discovered at swimming pool"

in_2 = "Sea otter spotted on surfboard by beach"

in_3 = "Baby panda enjoys boat ride"


in_4 = "Breakfast themed food truck beloved by all!"

in_5 = "New curry restaurant aims to please!"


in_6 = "Python developers are wonderful people"

in_7 = "TypeScript, C++ or Java? All are great!" 


input_text_lst_news = [in_1, in_2, in_3, in_4, in_5, in_6, in_7]

import numpy as np
from vertexai.language_models import TextEmbeddingModel

embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

embeddings = []
for input_text in input_text_lst_news:
    emb = embedding_model.get_embeddings([input_text])[0].values
    embeddings.append(emb)

embeddings_array = np.array(embeddings)

### Reduce embeddings from 768 to 2 dimensions for visualization
from sklearn.decomposition import PCA

PCA_model = PCA(n_components = 2)

PCA_model.fit(embeddings_array)

new_values = PCA_model.transform(embeddings_array)

import matplotlib.pyplot as plt

from utils import plot_2D
plot_2D(new_values[:,0], new_values[:,1], input_text_lst_news)

### Embbedings and similarity

in_1 = """He couldn’t desert 
          his post at the power plant."""

in_2 = """The power plant needed 
          him at the time."""

in_3 = """Cacti are able to 
          withstand dry environments.""" 

in_4 = """Desert plants can 
          survive droughts.""" 

input_text_lst_sim = [in_1, in_2, in_3, in_4]

embeddings = []
for input_text in input_text_lst_sim:
    emb = embedding_model.get_embeddings([input_text])[0].values
    embeddings.append(emb)
    
embeddings_array = np.array(embeddings) 

from utils import plot_heatmap

y_labels = input_text_lst_sim

# Plot the heatmap
plot_heatmap(embeddings_array, y_labels = y_labels, title = "Embeddings Heatmap")

from sklearn.metrics.pairwise import cosine_similarity

def compare(embeddings,idx1,idx2):
    return cosine_similarity([embeddings[idx1]],[embeddings[idx2]])

print(in_1)
print(in_2)
print(compare(embeddings,0,1))

print(in_1)
print(in_4)
print(compare(embeddings,0,3))
