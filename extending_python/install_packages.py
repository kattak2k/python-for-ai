
# bs4 is short for Beautiful Soup 4.
# !pip install bs4
#  !pip install aisetup
from bs4 import BeautifulSoup

import requests # let's you download webpages into python
from helper_functions import * 
from IPython.display import HTML, display

# The url from one of the Batch's newsletter
url = 'https://www.deeplearning.ai/the-batch/the-world-needs-more-intelligence/'
# Getting the content from the webpage's contents
response = requests.get(url)
print(HTML(f'<iframe src={url} width="60%" height="400"></iframe>'))



# -----TExt extraction from the webpage----

# Using beautifulsoup to extract the text
soup = BeautifulSoup(response.text, 'html.parser')
# Find all the text in paragraph elements on the webpage
all_text = soup.find_all('p')
# Create an empty string to store the extracted text
combined_text = ""
# Iterate over 'all_text' and add to the combined_text string
for text in all_text:
    combined_text = combined_text + "\n" + text.get_text()
# Print the final combined text
print(combined_text)
