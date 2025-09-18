import os

from langchain.llms import OpenAI,HuggingFaceHub

os.environ["OPENAI_API_KEY"] = "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"

###### Basic LLMChain #######
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

llm = OpenAI(model_name = "text-davinci-003",temperature = 0.0,max_tokens = 256)

article = '''Coinbase, the second-largest crypto exchange by trading volume, released its Q4 2022 earnings on Tuesday, giving shareholders and market players alike an updated look into its financials. In response to the report, the company's shares are down modestly in early after-hours trading.In the fourth quarter of 2022, Coinbase generated $605 million in total revenue, down sharply from $2.49 billion in the year-ago quarter. Coinbase's top line was not enough to cover its expenses: The company lost $557 million in the three-month period on a GAAP basis (net income) worth -$2.46 per share, and an adjusted EBITDA deficit of $124 million.Wall Street expected Coinbase to report $581.2 million in revenue and earnings per share of -$2.44 with adjusted EBITDA of -$201.8 million driven by 8.4 million monthly transaction users (MTUs), according to data provided by Yahoo Finance.Before its Q4 earnings were released, Coinbase's stock had risen 86% year-to-date. Even with that rally, the value of Coinbase when measured on a per-share basis is still down significantly from its 52-week high of $206.79.That Coinbase beat revenue expectations is notable in that it came with declines in trading volume; Coinbase historically generated the bulk of its revenues from trading fees, making Q4 2022 notable. Consumer trading volumes fell from $26 billion in the third quarter of last year to $20 billion in Q4, while institutional volumes across the same timeframe fell from $133 billion to $125 billion.The overall crypto market capitalization fell about 64%, or $1.5 trillion during 2022, which resulted in Coinbase's total trading volumes and transaction revenues to fall 50% and 66% year-over-year, respectively, the company reported.As you would expect with declines in trading volume, trading revenue at Coinbase fell in Q4 compared to the third quarter of last year, dipping from $365.9 million to $322.1 million. (TechCrunch is comparing Coinbase's Q4 2022 results to Q3 2022 instead of Q4 2021, as the latter comparison would be less useful given how much the crypto market has changed in the last year; we're all aware that overall crypto activity has fallen from the final months of 2021.)There were bits of good news in the Coinbase report. While Coinbase's trading revenues were less than exuberant, the company's other revenues posted gains. What Coinbase calls its "subscription and services revenue" rose from $210.5 million in Q3 2022 to $282.8 million in Q4 of the same year, a gain of just over 34% in a single quarter.And even as the crypto industry faced a number of catastrophic events, including the Terra/LUNA and FTX collapses to name a few, there was still growth in other areas. The monthly active developers in crypto have more than doubled since 2020 to over 20,000, while major brands like Starbucks, Nike and Adidas have dived into the space alongside social media platforms like Instagram and Reddit.With big players getting into crypto, industry players are hoping this move results in greater adoption both for product use cases and trading volumes. Although there was a lot of movement from traditional retail markets and Web 2.0 businesses, trading volume for both consumer and institutional users fell quarter-over-quarter for Coinbase.Looking forward, it'll be interesting to see if these pieces pick back up and trading interest reemerges in 2023, or if platforms like Coinbase will have to keep looking elsewhere for revenue (like its subscription service) if users continue to shy away from the market.
'''

fact_extraction_prompt = PromptTemplate(input_variables=["text_input"],template = "Extract the key facts out of this text. Don't include opinions. Give each fact a number and keep them short sentences. :\n\n {text_input}")

fact_extraction_chain = LLMChain(llm=llm,prompt=fact_extraction_prompt)

facts = fact_extraction_chain.run(article)

print(facts)

# Rewrite as a summary from the facts

investor_update_prompt = PromptTemplate(input_variables=["facts"],template="You are a Goldman Sachs analyst. Take the following list of facts and use them to write a short paragrah for investors. Don't leave out key info:\n\n {facts}")

investor_update_chain = LLMChain(llm=llm,prompt=investor_update_prompt)

investor_update = investor_update_chain.run(facts)

print(investor_update)

triples_prompt = PromptTemplate(input_variables=["facts"],template="Take the following list of facts and turn them into triples for a knowledge graph:\n\n {facts} ")

triples_chain = LLMChain(llm=llm,prompt=triples_prompt)

triples = triples_chain.run(facts)

print(triples)

########## Chaising these together
from langchain.chains import SimpleSequentialChain,SequentialChain

full_chain = SimpleSequentialChain(chains=[fact_extraction_chain,investor_update_chain],verbose = True)

response = full_chain.run(article)

###### PAL Math Chain
from langchain.chains import PALChain

llm = OpenAI(model_name = "code-davinci-002",temperature = 0, max_tokens = 512)

pal_chain = PALChain.from_math_prompt(llm,verbose = True)

question = "Jan has three times the number of pets as Marcia. Marcia has two more pets than Cindy. If Cindy has four pets, how many total pets do the three have?"

question_02= "The cafeteria had 23 apples. If they used 20 for lunch and bought 6 more, how many apples do they have?"

pal_chain.run(question_02)

######### API Chains
from langchain.chains.api.prompt import API_RESPONSE_PROMPT
from langchain.chains.api import open_meteo_docs
from langchain.chains import APIChain
from langchain.prompts.prompt import PromptTemplate

llm = OpenAI(temperature=0,max_tokens = 100)

chain_new = APIChain.from_llm_and_api_docs(llm,open_meteo_docs.OPEN_METEO_DOCS,verbose= True)

chain_new.run("What is the temperature like right now  in Bedok, Singapore in degrees Celcius?")

chain_new.run("Is it raining in Singapore?")
