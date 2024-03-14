from groq import Groq
import os
import time
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()

db=FAISS.load_local('tlds.faiss', embeddings, allow_dangerous_deserialization=True)

os.environ['GROQ_API_KEY'] = "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"
client = Groq()

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
            "person-name": 0,
            "domains": {
            "count": 10,
            "domains": [
                {
                "domain": "ApexSoftware.engineer",
                "sld": "ApexSoftware",
                "tld": "engineer",
                "tldPhase": "GA",
                "exact": 0,
                "registered": 0,
                "premium": 0,
                "adult": 0,
                "sldValue": 1.0,
                "tldValue": 1.0,
                "score": 1.0
                },
                {
                "domain": "ApexSoftware.engineering",
                "sld": "ApexSoftware",
                "tld": "engineering",
                "tldPhase": "GA",
                "exact": 0,
                "registered": 0,
                "premium": 0,
                "adult": 0,
                "sldValue": 1.0,
                "tldValue": 0.94,
                "score": 0.94
                },
                {
                "domain": "ApexSoftware.download",
                "sld": "ApexSoftware",
                "tld": "download",
                "tldPhase": "GA",
                "exact": 0,
                "registered": 0,
                "premium": 0,
                "adult": 0,
                "sldValue": 1.0,
                "tldValue": 0.84,
                "score": 0.84
                },
                {
                "domain": "ApexSoftware.tech",
                "sld": "ApexSoftware",
                "tld": "tech",
                "tldPhase": "GA",
                "exact": 0,
                "registered": 0,
                "premium": 0,
                "adult": 0,
                "sldValue": 1.0,
                "tldValue": 0.65,
                "score": 0.65
                },
                {
                "domain": "ApexSoftware.technology",
                "sld": "ApexSoftware",
                "tld": "technology",
                "tldPhase": "GA",
                "exact": 0,
                "registered": 0,
                "premium": 0,
                "adult": 0,
                "sldValue": 1.0,
                "tldValue": 0.64,
                "score": 0.64
                '},
                '{
                "domain": "ApexSoftware.reviews",
                "sld": "ApexSoftware",
                "tld": "reviews",
                "tldPhase": "GA",
                "exact": 0,
                "registered": 0,
                "premium": 0,
                "adult": 0,
                "sldValue": 1.0,
                "tldValue": 0.56,
                "score": 0.56
                },
                {
                "domain": "Apex-Software.engineer",
                "sld": "Apex-Software",
                "tld": "engineer",
                "tldPhase": "GA",
                "exact": 0,
                "registered": 0,
                "premium": 0,
                "adult": 0,
                "sldValue": 0.5,
                "tldValue": 1.0,
                "score": 0.5
                },
                {
                "domain": "Apex-Software.engineering",
                "sld": "Apex-Software",
                "tld": "engineering",
                "tldPhase": "GA",
                "exact": 0,
                "registered": 0,
                "premium": 0,
                "adult": 0,
                "sldValue": 0.5,
                "tldValue": 0.94,
                "score": 0.47
                },
                {
                "domain": "Apex-Software.download",
                "sld": "Apex-Software",
                "tld": "download",
                "tldPhase": "GA",
                "exact": 0,
                "registered": 0,
                "premium": 0,
                "adult": 0,
                "sldValue": 0.5,
                "tldValue": 0.84,
                "score": 0.42
                },
                {
                "domain": "ApexSoftware.computer",
                "sld": "ApexSoftware",
                "tld": "computer",
                "tldPhase": "GA",
                "exact": 0,
                "registered": 0,
                "premium": 0,
                "adult": 0,
                "sldValue": 1.0,
                "tldValue": 0.42,
                "score": 0.42
                }
            ]
            }
            }
            }

            Only return the JSON in proper format and nothing else.
            """

def prompt_builder(keyword):
    tlds = [d.page_content for d in db.similarity_search(keyword, k=5)]
    system_prompt = prompt_1 + ", ".join(tlds) +"\n\n"+ prompt_2
    print(system_prompt)
    return [
        {
            "role": "system",
            "content":  system_prompt 
            
        },
        {
            "role": "user",
            "content": keyword
        }
    ]

def get_groq_response(keyword):
    

    prompt = prompt_builder(keyword)
    completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=prompt,
        temperature=0,
        max_tokens=4160,
        top_p=1,
        stream=False,
        stop=None,
    )
    

    return completion.choices[0].message.content
