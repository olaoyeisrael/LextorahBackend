import asyncio
import httpx

async def test_email():
    url = "https://lextorah-elearning.com/elearning/api/send-email"
    headers = {
        "Content-Type": "application/json",
        "X-Internal-Token": "75926e792cd2c1f6d59be6097ab3bdce116dfd8eb9bf678616cb65351b805437"
    }
    payload = {
        "email": "olaoyeaisrael@gmail.com",
        "subject": "Test from Backend Script",
        "body": "<p>Test</p>",
        "greeting": "Hello,",
        "action_text": "View",
        "action_url": "https://lextorah-elearning.com",
        "queue": True
    }
    print("Sending POST request to", url)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=10.0)
            print("Status Code:", response.status_code)
            print("Response:", response.text)
    except Exception as e:
        print("Exception:", e)

if __name__ == "__main__":
    asyncio.run(test_email())
