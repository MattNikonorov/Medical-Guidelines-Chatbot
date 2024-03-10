from flask import Flask, request, jsonify
from flask import Flask, render_template, request, url_for

from llama_index import SimpleDirectoryReader, GPTListIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os
import time
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas
import openai
import numpy as np
import glob
import datetime


print(int(time.time()))

os.environ["OPENAI_API_KEY"] = 'API_KEY'

openai.api_key = 'API_KEY'

ips = []
ips_times = []

ips_ref = []
ips_times_ref = []

# To split our transcript into pieces
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Our chat model. We'll use the default which is gpt-3.5-turbo
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

# Prompt templates for dynamic values
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate, # I included this one so you know you'll have it but we won't be using it
    HumanMessagePromptTemplate
)

# To create our chat messages
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate


app = Flask(__name__)
@app.route("/")
def home():
    return render_template("bot.html")



llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")


"""

gfiles = glob.glob("guideline_files/pdftotext/*")

print(gfiles)
for g1 in range(33, 34, 1):
    print(g1)


    f = open(f"embs_ult{g1}.csv", "w")
    f.write("combined")
    f.close()

    content = ""
    with open(f"{gfiles[g1]}", 'r') as file:
        content += file.read()
        content += "\n\n"

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=2000, chunk_overlap=250)
    texts = text_splitter.split_text(content)


    def get_embedding(text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

    df = pandas.read_csv(f"embs_ult{g1}.csv")



    df["combined"] = texts
    for i4 in range(len(df["combined"])):
        df["combined"][i4] = '""' + df["combined"][i4].replace("\n", "") + '""'
    df.to_csv(f"embs_ult{g1}.csv")

    df["embedding"] = df.combined.apply(lambda x: get_embedding(x))
    df.to_csv(f"embs_ult{g1}.csv", index=False)

    df = pandas.read_csv(f"embs_ult{g1}.csv")

    embs = []
    for r1 in range(len(df.embedding)):
        #print(df.embedding[r1].split(","))
        #print(type(df.embedding[r1].split(",")))
        e1 = df.embedding[r1].split(",")
        for ei2 in range(len(e1)):
            e1[ei2] = float(e1[ei2].strip().replace("[", "").replace("]", ""))
        embs.append(e1)

    df["embedding"] = embs
    df.to_csv(f"embs_ult{g1}.csv", index=False)

"""

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']



def logic(question):
    
    df = pandas.read_csv(f"embs_fold/embs_ult_f.csv")

    embs = []
    for r1 in range(len(df.embedding)):
        #print(df.embedding[r1].split(","))
        #print(type(df.embedding[r1].split(",")))
        e1 = df.embedding[r1].split(",")
        for ei2 in range(len(e1)):
            e1[ei2] = float(e1[ei2].strip().replace("[", "").replace("]", ""))
        embs.append(e1)

    df["embedding"] = embs

    bot_message = ""
    product_embedding = get_embedding(
        question
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))
    df.to_csv("embs_fold/embs_ult_f.csv")

    df2 = df.sort_values("similarity", ascending=False)
    df2.to_csv("embs_fold/embs_ult_f.csv")
    df2 = pandas.read_csv("embs_fold/embs_ult_f.csv")
    print(df2["similarity"][0])

    from langchain.docstore.document import Document

    comb = [df2["combined"][0]]
    docs = [Document(page_content=t) for t in comb]
    #print(docs)

    template="""
    You are a helpful assistant that has deep knowledge on clinical guidelines from your dataset and can advise what diagnostics or therapy including line of therapy and possible drug combinations recommended for specific patient type (disease stage and etc…).

    You will provide answers with references from the guideline. You can also provide insights on what studies and publications were used to inform guidelines.

    Do not respond outside of the document provided. If you are not sure, say you do not know or such data does not exist in your dataset. Always provide relevant references to the guidelines
    """

    prompt_template = template + "\n" + question + """

    {text}

    """ 

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)

    output = chain.run(docs)

    return output





@app.route('/chat', methods=['POST'])

def chat():

    user_message = request.json['message']
    pm = request.json['pm']

    ips.append(request.remote_addr)
    ips_times.append(int(time.time()))

    ips_ref.append(request.remote_addr)
    ips_times_ref.append(int(time.time()))

    for ipi in range(len(ips_ref)):
        if ips_times_ref[ipi] < ips_times_ref[-1]-86400:
            ips.remove(ips_ref[ipi])
            ips_times.remove(ips_times_ref[ipi])


    if ips.count(request.remote_addr) > 50:
        return jsonify({'message': "request limit"})
    else:
        response = logic(user_message)
        if ("sorry" in response.lower()) or ("provide more" in response.lower()) or ("not found" in response.lower()) or ("does not mention" in response.lower()) or ("does not reference" in response.lower()) or ("no information" in response.lower()) or ("not enough information" in response.lower()) or ("unable to provide" in response.lower()) or ("the guidelines do not" in response.lower()):
            response = logic(str(pm + ' ' + user_message))

        response = response.replace("<", "").replace(">", "")
        return jsonify({'message': response})


if __name__ == "__main__":
    app.run(host="localhost", port=8001, debug=True)
