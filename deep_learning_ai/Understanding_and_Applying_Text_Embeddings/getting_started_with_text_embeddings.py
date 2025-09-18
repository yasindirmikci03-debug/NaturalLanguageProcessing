from utils import authenticate

credentials, PROJECT_ID = authenticate()

REGION = 'us-central1'

import vertexai

vertexai.init(project= PROJECT_ID,location = REGION,credentials=credentials)

from vertexai.language_models import TextEmbeddingModel

embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

embedding = embedding_model.get_embeddings(["lige"])

vector = embedding[0].values
print(vector[:10])

embedding = embedding_model.get_embeddings(["What is the meaning of life?"])

vector = embedding[0].values
print(vector[:10])

# Similarity
from sklearn.metrics.pairwise import cosine_similarity

emb_1 = embedding_model.get_embeddings(["What is the meaning of life?"])

emb_2 = embedding_model.get_embeddings(["How does one spend their time well on Earth?"])

emb_3 = embedding_model.get_embeddings(["Would you like a salad?"])

vec_1 = [emb_1[0].values]
vec_2 = [emb_2[0].values]
vec_3 = [emb_3[0].values]

print(cosine_similarity(vec_1,vec_2)) 
print(cosine_similarity(vec_2,vec_3))
print(cosine_similarity(vec_1,vec_3))

#### From word to sentence embeddings

in_1 = "The kids play in the park."
in_2 = "The play was for kids in the park."

in_pp_1 = ["kids","play","park"]
in_pp_2 = ["play","kids","park"]

embeddings_1 = [emb.values for emb in embedding_model.get_embeddings(in_pp_1)]

import numpy as np

emb_array_1 = np.stack(embeddings_1)

embeddings_2 = [emb.values for emb in embedding_model.get_embeddings(in_pp_2)]

emb_array_2 = np.stack(embeddings_2)

emb_1_mean = emb_array_1.mean(axis=0)

emb_2_mean = emb_array_2.mean(axis=0)

embedding_1 = embedding_model.get_embeddings([in_1])

embedding_2 = embedding_model.get_embeddings([in_2])

vector_1 = embedding_1[0].values
print(vector_1[:4])
vector_2 = embedding_2[0].values
print(vector_2[:4])