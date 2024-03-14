import streamlit as st
import requests
from mixtral_call import get_groq_response
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

def call_api_2(keyword):
    results = get_groq_response(keyword)
    try:
        json_data = json.loads(results)
    except ValueError:
        # The JSON data is invalid
        json_data = extract_json_from_string(results)
    return json_data

def main():
    st.title("API Results Comparison")
    keyword = st.text_input("Enter a keyword")

    if st.button("Get Results"):

        if keyword:
            col1, col2 = st.columns(2)
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
                try:
                    results_2 = json.loads(extract_json_from_string(call_api_2(keyword)))["output"]
                except:
                    results_2 = extract_json_from_string(call_api_2(keyword))                
                end_time = time.time()  # Stop measuring time
                response_time = end_time - start_time
                st.write(f"Results from GROQ Mixtral received in {response_time} seconds:")
                st.write(results_2)
        else:
            st.warning("Please enter a keyword")

if __name__ == "__main__":
    main()