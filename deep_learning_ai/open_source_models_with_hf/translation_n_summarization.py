import torch
import gc

from transformers.utils import logging
from transformers import pipeline

logging.set_verbosity_error()

translator = pipeline("translation",model = "facebook/nllb-200-distilled-600M",torch_dtype=torch.bfloat16)

text = """\
My puppy is adorable, \
Your kitten is cute.
Her panda is friendly.
His llama is thoughtful. \
We all have nice pets!"""

text_translated = translator(text,src_lang = "eng_Latn",tgt_lang = "fra_Latn" )

print(text_translated)

summarizer = pipeline("summarization",model = "facebook/bart-large-cnn",torch_dtype=torch.bfloat16)

text2 = """Paris is the capital and most populous city of France, with
          an estimated population of 2,175,601 residents as of 2018,
          in an area of more than 105 square kilometres (41 square
          miles). The City of Paris is the centre and seat of
          government of the region and province of Île-de-France, or
          Paris Region, which has an estimated population of
          12,174,880, or about 18 percent of the population of France
          as of 2017."""

text_summarized = summarizer(text2,min_length = 10,max_length= 100)
print(text_summarized)