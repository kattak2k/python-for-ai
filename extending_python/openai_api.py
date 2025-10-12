
# !pip install openai
# pip3 --version install aisetup install openai
import os
from dotenv import load_dotenv
from openai import OpenAI

# Get the OpenAI API key from the .env file
load_dotenv('.env', override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key = openai_api_key)

# def get_llm_response(prompt):
#     completion = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are an AI assistant.",
#             },
#             {"role": "user", "content": prompt},
#         ],
#         temperature=0.0,
#     )
#     response = completion.choices[0].message.content
#     return response

prompt = "What is the capital of France?"
response = get_llm_response(prompt)
print(response)


# --------
from aisetup import authenticate, print_llm_response, get_llm_response
from dotenv import load_dotenv
import os

load_dotenv('.env', override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
authenticate(openai_api_key)

# Print the LLM response
print_llm_response("What is the capital of France")

# Store the LLM response as a variable and then print
response = print_llm_response("What is the capital of France")
print(response)