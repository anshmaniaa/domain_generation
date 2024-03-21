
import boto3
import os
import time
import json
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()

db=FAISS.load_local('tlds.faiss', embeddings, allow_dangerous_deserialization=True)

def create_bedrock_client():
    client = boto3.client(service_name='bedrock-runtime')

    # print(client.list_foundation_models())
    return client

client = create_bedrock_client()

prompt_1 = """
            You are a domain recommendation system.
            Your objective is to recommend a domain name based on the user's input.
            The domain names should be short, memorable, and relevant to the user's input.
            
            You can reshuffle, shorten, use synonyms, hyphens to the user's input to generate domain name suggestions.

            The number of words in the domain name should be less than or equal to user query.

            Only use the following TLDs for your suggestions if they make contextual sense else use generic TLDs like com, org, live: """
prompt_2 = """
            Here is an example of how to return your results in the following JSON format:
            {
            "output": {
            "count": 10,
            "domains": 
                ["domain": "ApexSoftware.engineer",

                "domain": "ApexSoftware.engineering",
            
                "domain": "ApexSoftware.download",
              
                "domain": "ApexSoftware.tech",
                
                "domain": "ApexSoftware.technology",
                
                "domain": "ApexSoftware.reviews",
               
                "domain": "Apex-Software.engineer",
                
                "domain": "Apex-Software.engineering",
                
                "domain": "Apex-Software.download",
                
                "domain": "ApexSoftware.computer",
                ]
                
            }
            }

            Only return the JSON in proper format and nothing else.
            """

def prompt_builder(keyword):
    tlds = [d.page_content for d in db.similarity_search(keyword, k=5)]
    system_prompt = prompt_1 + ", ".join(tlds) +"\n\n"+ prompt_2
    # print(system_prompt)
    system= "<s>[INST] " + system_prompt + " [INST]</s> "
    user = "<s>[USER] " + keyword + " [USER]</s>"
    return system  + user


def get_bedrock_response(keyword):
    prompt = prompt_builder(keyword)

    body = json.dumps({
        "prompt": prompt,
        # "max_tokens": 4160,
        "temperature": 0,
        "top_p": 1,
        "top_k": 50
    })


    completion = client.invoke_model(
        body= body, 
        modelId="mistral.mixtral-8x7b-instruct-v0:1", 
        accept = "application/json",
        contentType = "application/json"
    )
    
    body_content = completion.get('body').read()
    # print("response body content",body_content,type(body_content))
    # return json.loads(body_content
    return  body_content
