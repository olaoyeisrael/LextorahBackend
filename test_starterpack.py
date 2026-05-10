import asyncio
import httpx

async def test():
    url = "http://127.0.0.1:8000/starterpack"
    payload = {
        "institutionName": "Test Institution",
        "email": "test@example.com"
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=payload, timeout=20.0)
            print("Status:", resp.status_code)
            print("Body:", resp.text)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    asyncio.run(test())
