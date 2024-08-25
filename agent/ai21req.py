import os
import requests
import json
from ai21.studio.ai21api import AI21Client, ChatMessage

def process_get_response(response, headers=None):
    """
    Sends a GET request to the specified URL, processes the response,
    and uses AI21 to generate a completion based on the response data.
    
    Args:
    url (str): URL to send the GET request to.
    headers (dict, optional): Headers to include in the request.
    
    Returns:
    str: The response from AI21Client.
    """
    # Send the GET request
    response = requests.get(url, headers=headers if headers else {})

    # Convert the response text to a JSON string
    data_json = json.dumps(response.text)

    # Print the JSON string (for debugging)
    print(data_json)

    # Initialize AI21Client with the API key
    client = AI21Client(api_key=os.environ.get("AI21_API_KEY"))

    # Create a completion with AI21 using the response data
    response_a = client.chat.completions.create(
        model="jamba-instruct-preview",
        messages=[ChatMessage(
            content="find pii in:" + str(data_json),
            role="user",
        )],
        n=1, 
        max_tokens=1024,
        temperature=0.7,
        top_p=1,
        stop=[],
    )

    # Print the completion (for debugging)
    print(response_a)

    # Return the formatted completion response or the full object as needed
    return response_a.text if hasattr(response_a, 'text') else str(response_a)