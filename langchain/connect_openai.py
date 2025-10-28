import os
from openai import OpenAI 

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
#openai.api_key = os.environ['OPENAI_API_KEY']

#Note: LLM's do not always produce the same results. When executing the code in your notebook, you may get slightly different answers that those in the video.

# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

## Chat API : OpenAI

## Let's start with a direct API calls to OpenAI.
client = OpenAI(
    # This is the default and can be omitted
   api_key=os.environ.get("OPENAI_API_KEY"),
   organization=os.environ.get("ORGANIZATION_ID")
)


def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    return response.choices[0].message["content"]


response = get_completion("What is 1+1?")

# customer_email = """
# Arrr, I be fuming that me blender lid \
# flew off and splattered me kitchen walls \
# with smoothie! And to make matters worse,\
# the warranty don't cover the cost of \
# cleaning up me kitchen. I need yer help \
# right now, matey!
# """

# style = """American English \
# in a calm and respectful tone
# """

# prompt = f"""Translate the text \
# that is delimited by triple backticks 
# into a style that is {style}.
# text: ```{customer_email}```
# """

# print(prompt)

# response = get_completion(prompt)

print(response)
