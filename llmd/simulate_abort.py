import asyncio
import httpx


API_URL = "http://localhost:40876/v1/completions" 
# API_URL = "http://localhost:8000/v1/completions" 
NUM_REQUESTS = 10
ABORT_INDICES = {1, 9} 

async def send_request(i, abort=False):
    async with httpx.AsyncClient(timeout=None) as client:
        task = asyncio.create_task(client.post(
            API_URL,
            json={
                "model": "Qwen/Qwen3-0.6B",
                "prompt": f"This is test request {i}. what do you say",
                "max_tokens": 100,
            },
            timeout=None
        ))

        if abort:
            # await asyncio.sleep(0.2 + random.random() * 0.3)  
            await asyncio.sleep(0.5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                print(f"Request {i} aborted!!\n")
        else:
            try:
                response = await task
                print(f"Request {i} completed: {response.status_code}")
            except Exception as e:
                print(f"Request {i} failed: {e}")

async def main():
    tasks = [
        send_request(i, abort=(i in ABORT_INDICES))
        for i in range(NUM_REQUESTS)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

asyncio.run(main())
