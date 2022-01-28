#!/usr/bin/env python

import asyncio
import websockets

async def hello():
    async with websockets.connect("ws://localhost:8001") as websocket:
        await websocket.send("Hello world!")
        message = await websocket.recv()
        print(message)

asyncio.run(hello())
