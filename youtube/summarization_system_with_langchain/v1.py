from langchain import OpenAI,PromptTemplate,LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chains import LLMChain
import textwrap

llm = OpenAI()

text_splitter = CharacterTextSplitter()

with open('/content/how_to_win_friends.txt') as f:
    how_to_win_friends = f.read()

texts = text_splitter.split_text(how_to_win_friends)

chain = load_summarize_chain(llm, chain_type= "map_reduce")

docs = [Document(page_content=t) for t in texts[:4]]

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary,width=100)
print(wrapped_text)

# for summarizing each part 
chain.llm_chain.prompt_template

# for combining the parts
chain.combine_documents_chain.llm_chain.prompt_template

chain = load_summarize_chain(llm, chain_type= "map_reduce",verbose=True)
output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary,width=100,break_long_words=False,replace_whitespace=False)

# Summarizing with the 'stuff' chain

chain = load_summarize_chain(llm,chain_type="stuff")

prompt_template = """Write a concise bullet point summary of the following

{text}

CONSCISE SUMMARY IN BULLET POINTS:"""

BULLET_POINT_PROMPT = PromptTemplate(template = prompt_template,input_variables=['text'])

chain = load_summarize_chain(llm,chain_type="stuff",prompt = BULLET_POINT_PROMPT)

output_summary = chain.run(docs)

wrapped_text = textwrap.fill(output_summary,width=100,break_long_words=False,replace_whitespace=False)


#### With 'map_reduce' our custom prompt

chain = load_summarize_chain(llm,chain_type="map_reduce",prompt = BULLET_POINT_PROMPT,combine_prompt = BULLET_POINT_PROMPT) 

output_summary = chain.run(docs)

wrapped_text = textwrap.fill(output_summary,width=100,break_long_words=False,replace_whitespace=False)

# with intermediate steps

PROMPT = PromptTemplate(template=prompt_template,input_variables=['text'])

chain = load_summarize_chain(OpenAI(temperature = 0),
                            chain_type="map_reduce",
                            return_intermediate_steps = True,
                            map_prompt = PROMPT,
                            combine_prompt = PROMPT)

output_summary = chain({"input_documents" : docs}, return_only_outputs = True)

wrapped_text = textwrap.fill(output_summary['output_text'],width = 100,break_long_words=False,replace_whitespace=False)
print(wrapped_text)
wrapped_text = textwrap.fill(output_summary['intermediate_steps'][2],width = 100,break_long_words=False,replace_whitespace=False)
print(wrapped_text)



#### PART 2

llm = OpenAI(model_name = "text-davinci-003",temperature =0, max_tokens = 256)

article = '''Coinbase, the second-largest crypto exchange by trading volume, released its Q4 2022 earnings on Tuesday, giving shareholders and market players alike an updated look into its financials. In response to the report, the company's shares are down modestly in early after-hours trading.In the fourth quarter of 2022, Coinbase generated $605 million in total revenue, down sharply from $2.49 billion in the year-ago quarter. Coinbase's top line was not enough to cover its expenses: The company lost $557 million in the three-month period on a GAAP basis (net income) worth -$2.46 per share, and an adjusted EBITDA deficit of $124 million.Wall Street expected Coinbase to report $581.2 million in revenue and earnings per share of -$2.44 with adjusted EBITDA of -$201.8 million driven by 8.4 million monthly transaction users (MTUs), according to data provided by Yahoo Finance.Before its Q4 earnings were released, Coinbase's stock had risen 86% year-to-date. Even with that rally, the value of Coinbase when measured on a per-share basis is still down significantly from its 52-week high of $206.79.That Coinbase beat revenue expectations is notable in that it came with declines in trading volume; Coinbase historically generated the bulk of its revenues from trading fees, making Q4 2022 notable. Consumer trading volumes fell from $26 billion in the third quarter of last year to $20 billion in Q4, while institutional volumes across the same timeframe fell from $133 billion to $125 billion.The overall crypto market capitalization fell about 64%, or $1.5 trillion during 2022, which resulted in Coinbase's total trading volumes and transaction revenues to fall 50% and 66% year-over-year, respectively, the company reported.As you would expect with declines in trading volume, trading revenue at Coinbase fell in Q4 compared to the third quarter of last year, dipping from $365.9 million to $322.1 million. (TechCrunch is comparing Coinbase's Q4 2022 results to Q3 2022 instead of Q4 2021, as the latter comparison would be less useful given how much the crypto market has changed in the last year; we're all aware that overall crypto activity has fallen from the final months of 2021.)There were bits of good news in the Coinbase report. While Coinbase's trading revenues were less than exuberant, the company's other revenues posted gains. What Coinbase calls its "subscription and services revenue" rose from $210.5 million in Q3 2022 to $282.8 million in Q4 of the same year, a gain of just over 34% in a single quarter.And even as the crypto industry faced a number of catastrophic events, including the Terra/LUNA and FTX collapses to name a few, there was still growth in other areas. The monthly active developers in crypto have more than doubled since 2020 to over 20,000, while major brands like Starbucks, Nike and Adidas have dived into the space alongside social media platforms like Instagram and Reddit.With big players getting into crypto, industry players are hoping this move results in greater adoption both for product use cases and trading volumes. Although there was a lot of movement from traditional retail markets and Web 2.0 businesses, trading volume for both consumer and institutional users fell quarter-over-quarter for Coinbase.Looking forward, it'll be interesting to see if these pieces pick back up and trading interest reemerges in 2023, or if platforms like Coinbase will have to keep looking elsewhere for revenue (like its subscription service) if users continue to shy away from the market.
'''

wrapped_text = textwrap.fill(article, 
                             width=100,
                             break_long_words=False,
                             replace_whitespace=False)
print(wrapped_text)

fact_extraction_prompt = PromptTemplate(input_variables=["text_input"],template="Extract the key facts of this text. Don't include opinions. Give each fact a number and keep them short sentences. :\n\n {text_input}")

fact_extraction_chain = LLMChain(llm=llm,prompt = fact_extraction_prompt)

facts = fact_extraction_chain.run(article)

wrapped_text = textwrap.fill(facts,width=100,break_long_words=False,replace_whitespace=False)

print(wrapped_text)

# Summarization Checking

from langchain.chains import LLMSummarizationCheckerChain

checker_chain = LLMSummarizationCheckerChain(llm=llm,verbose=True,max_checks=2)

final_summary = checker_chain.run(article)

print(final_summary)

checker_chain.create_assertions_prompt.template

# 'Given some text, extract a list of facts from the text. \n\n Format your output as a bulleted list.\n\nText:\n"""\n{summary}\n"""\n\nFacts:'

# Making triples to compare to a graph

triples_prompt = PromptTemplate(
    input_variables=["facts"],
    template = "Take the following list of facts and turn them into triples for a knowledge graph:\n\n {facts}"
)

triples_chain = LLMChain(llm=llm,prompt = triples_prompt)

triples = triples_chain.run(facts)

print(triples)


### Part 3 
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,AIMessagePromptTemplate,HumanMessagePromptTemplate
from langchain.schema import AIMessage,HumanMessage,SystemMessage

chatGPT = ChatOpenAI(temperature = 0)

chatGPT = ([HumanMessage(content = "Translate this sentence from English to French. I love programming.")])

# Normal Summary

messages = [
    SystemMessage(content = "You are an expert at making strong factual summarizations. Take the article submitted by the user and produce a factual useful summary"),
    HumanMessage(content = article)
]

responses = chatGPT(messages)

wrapped_text = textwrap.fill(responses.content,width=100,break_long_words=False,replace_whitespace=False)

print(wrapped_text)

# Bullet List Summary

messages = [
    SystemMessage(content = "You are an expert at making strong factual summarizations.Extract the key facts out of this text. Don't include opinions. Give each fact a number and keep them short sentences. :\n\n"),
    HumanMessage(content = article)
]

responses = chatGPT(messages)

wrapped_text = textwrap.fill(responses.content,width=100,break_long_words=False,replace_whitespace=False)

print(wrapped_text)

