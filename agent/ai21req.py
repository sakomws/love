import os
from ai21 import AI21Client
from ai21.models.chat import ChatMessage
import requests
import json
# Define the URL and the headers

#curl --request GET \
#  --url https://api.cloudflare.com/client/v4/accounts/abbe3864d6e415d5692f79c3964262e8/ai-gateway/gateways/phlanx/logs \
#  --header 'Authorization: Bearer 5y3vheaDFrZ-4A20fpIkM1m7m1_buscEaLMFECZZ' \
#  --header 'Content-Type: application/json'


url = "https://api.cloudflare.com/client/v4/accounts/abbe3864d6e415d5692f79c3964262e8/ai-gateway/gateways/phlanx/logs"
headers = {
    'Authorization': 'Bearer 5y3vheaDFrZ-4A20fpIkM1m7m1_buscEaLMFECZZ',  # Replace 'your_access_token' with your actual access token
    'Content-Type': 'application/json'
}

# Send the GET request
response = requests.get(url, headers=headers)

data_json = json.dumps(response.text)

# Print the JSON string
print(data_json)

client = AI21Client(
    # This is the default and can be omitted
    api_key=os.environ.get("AI21_API_KEY"),
)

response_a=client.chat.completions.create(
  model="jamba-instruct-preview",
  messages=[ChatMessage(
      content="find pii in:"+str(data_json),
      role="user",
    )
],
  n=1, 
  max_tokens=1024,
  temperature=0.7,
  top_p=1,
  stop=[],
)

print(response_a)