"""Asynchronous parsing"""
import asyncio
import random
import time

import httpx
from fastapi import FastAPI

app = FastAPI()

@app.post("/parse_url/")
async def parse_url(url: str) -> str:
    """Single web-page parsing (async version)"""
    try:
        async with httpx.AsyncClient(app=app) as client:
            r = await client.get(url)
            r.raise_for_status()

            parse_time = 0.1 * random.randint(5, 10) if random.random() < 0.1 else 0.1
            await asyncio.sleep(parse_time)

            return f"Parsed {url}"
    except Exception as e:
        return f"Error fetching {url}: {e}"

async def run_test(n_requests: int) -> float:
    """Parsing test simulation (async version)"""
    url = "https://httpbin.org/"
    tasks = []

    #with TestClient(app) as client:
    async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
        ts = time.time()
        # concurrency loop
        for _ in range(n_requests):
            # list of tasks for concurrency
            tasks.append(asyncio.ensure_future(client.post("/parse_url/", params={"url": url})))
            #_ = client.post("/parse_url/", params={"url": url})

        _ = await asyncio.gather(*tasks)
        return time.time() - ts

if __name__ == "__main__":
    t = asyncio.run(run_test(n_requests=10))
    print(f"Time taken: {t} seconds")
