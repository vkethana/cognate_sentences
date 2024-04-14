import requests
from bs4 import BeautifulSoup

def get_spanish_definition(word):
    url = f"https://www.wordreference.com/es/translation.asp?tranword={word}"
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Finding the first definition in the results
        definition = soup.find(class_='trans clickable').text.strip()
        return definition
    else:
        return "Failed to retrieve definition"

word = input("Enter a word in Spanish: ")
definition = get_spanish_definition(word)
print("Definition:", definition)

