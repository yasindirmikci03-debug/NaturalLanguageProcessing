from utils import authenticate

credentials, PROJECT_ID = authenticate()

REGION = 'us-central1'

import vertexai

vertexai.init(project= PROJECT_ID,location = REGION,credentials=credentials)

from google.cloud import bigquery
import pandas as pd

def run_bq_query(sql):

    # Create BQ Client
    bq_client = bigquery.Client(project= PROJECT_ID,credentials= credentials)

    # Try dry run before executing query to catch any errors
    job_config = bigquery.QueryJobConfig(dry_run = True, use_query_cache = False)

    bq_client.query(sql,job_config=job_config)

    client_result = bq_client.query(sql, 
                                    job_config=job_config)

    job_id = client_result.job_id

    # Wait for query/job to finish running. then get & return data frame
    df = client_result.result().to_arrow().to_pandas()
    print(f"Finished job_id: {job_id}")
    return df

language_list = ["python","html","r","css"]

so_df = pd.DataFrame()

for language in language_list:
    
    print(f"generating {language} dataframe")
    
    query = f"""
    SELECT
        CONCAT(q.title, q.body) as input_text,
        a.body AS output_text
    FROM
        `bigquery-public-data.stackoverflow.posts_questions` q
    JOIN
        `bigquery-public-data.stackoverflow.posts_answers` a
    ON
        q.accepted_answer_id = a.id
    WHERE 
        q.accepted_answer_id IS NOT NULL AND 
        REGEXP_CONTAINS(q.tags, "{language}") AND
        a.creation_date >= "2020-01-01"
    LIMIT 
        500
    """

    
    language_df = run_bq_query(query)
    language_df["category"] = language
    so_df = pd.concat([so_df, language_df], 
                      ignore_index = True) 
    

### Generate text embeddings
from vertexai.language_models import TextEmbeddingModel

model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")

import time 
import numpy as np

def generate_batches(sentences,batch_size = 5):
    for i in range(0,len(sentences),batch_size):
        yield sentences[i : i + batch_size]

so_questions = so_df[0:200].input_text.tolist()
batches = generate_batches(sentences = so_questions)

batch = next(batches)

### Get embeddings on a batch of data

def encode_texts_to_embeddings(sentences):
    try:
        embeddings = model.get_embeddings(sentences)
        return [embeddings.values for embedding in embeddings]
    except Exception:
        return [None for _ in range(len(sentences))]

batch_embeddings = encode_texts_to_embeddings(batch)

