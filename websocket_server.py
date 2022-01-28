#!/usr/bin/env python

import asyncio
import websockets
import json
from cv_track import remote_cv

async def echo(websocket):
    async for message in websocket:
        print(message)
        await websocket.send(message)

async def handler(websocket):
    while True:
        try:
            message = await websocket.recv()
        except websockets.ConnectionClosedOK:
            break
        print(message)

# async def handler2(websocket):
#     try:
#         print('handler2 called')
#         async for message in websocket:
#             print(message)
#             for player, column, row in [
#                 ('PLAYER1', 3, 0),
#                 ('PLAYER2', 3, 1),
#                 ('PLAYER3', 4, 0),
#                 ('PLAYER4', 4, 1),
#                 ('PLAYER5', 2, 0),
#                 ('PLAYER6', 1, 0),
#                 ('PLAYER7', 5, 0),
#             ]:
#                 event = {
#                     "type": "right",
#                     "player": player,
#                     "column": column,
#                     "row": row,
#                 }
#                 await websocket.send(json.dumps(event))
#                 await asyncio.sleep(0.5)
#     # event = {
#     #     "type": "q",
#     #     "player": 'PLAYER1',
#     # }
#     # await websocket.send(json.dumps(event))
#     except BaseException as err:
#         print(f"Unexpected {err=}, {type(err)=}")
#         raise

async def main():
    async with websockets.serve(remote_cv, "", 8002):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
    # my_cv()
