import streamlit as st
import requests
from mixtral_call import get_groq_response
from aws_mixtral_call import get_bedrock_response
from aws_mixtral_call_7b import get_bedrock_response_7b
import json
import time

api_1 = f"https://api.rns.domains/recommend-domains?account=default&password=*&input={input}"
def call_api_1(keyword):
    # Call API 1 and return the results
    # Replace <API_ENDPOINT_1> with the actual endpoint of API 1
    response = requests.get(api_1, params={"input": keyword})
    results = response.json()
    return results["output"]

def correct_json_string(json_string):
    # Count the number of opening and closing curly brackets
    opening_brackets = json_string.count("{")
    closing_brackets = json_string.count("}")
    
    # Calculate the difference in the number of brackets
    bracket_difference = opening_brackets - closing_brackets
    
    # Add missing closing brackets
    if bracket_difference > 0:
        json_string += "}" * bracket_difference
    
    # Add missing opening brackets
    elif bracket_difference < 0:
        json_string = "{" * abs(bracket_difference) + json_string
    
    return json_string

def extract_json_from_string(string):
    # Find the start and end indices of the JSON part
    start_index = string.find("{")
    end_index = string.rfind("}")
    
    # Extract the JSON part from the string
    json_part = string[start_index:end_index+1]
    
    return json_part

def extract_nested_json(json_str):
    try:
        # Parse the string as JSON
        outer_json = json.loads(json_str)
        
        # Access the 'text' field of the first element of the 'outputs' array
        inner_json_str = outer_json['outputs'][0]['text']
        
        # Parse this string as JSON
        inner_json = json.loads(inner_json_str)
        
        return inner_json
    except json.decoder.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"JSON string: {json_str}")

def call_api_2(keyword):
    results = get_groq_response(keyword)
    # try:
    #     json_data = json.loads(results)
    # except ValueError:
    #     # The JSON data is invalid
    #     json_data = extract_json_from_string(results)
    return results

def call_api_3(keyword):
    results = get_bedrock_response(keyword)
    print("aws bedrock results",results.decode('utf-8'))
   
    return results.decode('utf-8')


def call_api_4(keyword):
    results = get_bedrock_response_7b(keyword)
    print("aws bedrock results",results.decode('utf-8'))
   
    return results.decode('utf-8')

def main():
    st.title("API Results Comparison")
    keyword = st.text_input("Enter a keyword")

    if st.button("Get Results"):

        if keyword:
            col1, col2, col3,col4 = st.columns(4)
            with col1:
                st.write("Calling RNS...")
                start_time = time.time()  # Start measuring time
                results_1 = call_api_1(keyword)
                end_time = time.time()  # Stop measuring time
                response_time = end_time - start_time
                st.write(f"Results from RNS received in {response_time} seconds:")
                st.write(results_1)
            with col2:
                st.write("Calling GROQ...")
                start_time = time.time()  # Start measuring time
                # try:
                results_2 = call_api_2(keyword)
                # except:
                #     results_2 = extract_json_from_string(call_api_2(keyword))                
                end_time = time.time()  # Stop measuring time
                response_time = end_time - start_time
                st.write(f"Results from GROQ Mixtral received in {response_time} seconds:")
                st.write(results_2)
            with col3:
                st.write("Calling AWS Sagemaker mixtral 8x7b...")
                start_time = time.time()  # Start measuring time
                # try:
                results_3 = call_api_3(keyword)
                # except:
                #     results_2 = extract_nested_json(call_api_3(keyword))                
                end_time = time.time()  # Stop measuring time
                response_time = end_time - start_time
                st.write(f"Results from AWS sagemaker Mixtral received in {response_time} seconds:")
                st.write(results_3)
            with col4:
                st.write("Calling AWS Sagemaker mixtral 7b...")
                start_time = time.time()  # Start measuring time
                # try:
                results_3 = call_api_3(keyword)
                # except:
                #     results_2 = extract_nested_json(call_api_3(keyword))                
                end_time = time.time()  # Stop measuring time
                response_time = end_time - start_time
                st.write(f"Results from AWS sagemaker Mixtral received in {response_time} seconds:")
                st.write(results_3)
        else:
            st.warning("Please enter a keyword")

if __name__ == "__main__":
    main()