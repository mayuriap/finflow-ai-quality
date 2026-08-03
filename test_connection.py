from src.groq_client import GroqClient

client = GroqClient()
response = client.ask(
    'You are an AI testing assistant. Say hello and confirm you are ready to help with AI quality engineering in one sentence.'
)
print('API Test Result:')
print(response)