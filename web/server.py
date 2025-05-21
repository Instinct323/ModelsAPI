import asyncio
import logging
import pickle
import time

import fastapi

from simple_func import FUNCTIONS

# Config
FUNCTIONS: dict
CONNS_LIMIT = 1  # Limit the number of connections

# uvicorn server:app
LOGGER = logging.getLogger("uvicorn")
app = fastapi.FastAPI(
    title="Functions API",
    description="author: [TongZJ](https://github.com/Instinct323)\n\n" +
                "\n\n".join(f"## {k}\n\n```\n{v.__doc__ or ''}\n```" for k, v in FUNCTIONS.items()),
    version="1.0.0"
)
CONNS = set()
CONNS_LOCK = asyncio.Lock()


@app.post("/invoke")
async def invoke(request: fastapi.Request):
    """ Invoke a function on the server. """
    cost = {"send": time.time() - float(request.headers["t-send"])}
    # Limit the number of connections
    t0 = time.time()
    host = request.client.host
    async with CONNS_LOCK:
        if len(CONNS) < CONNS_LIMIT: CONNS.add(host)
        if host not in CONNS:
            raise fastapi.HTTPException(status_code=403, detail="Forbidden.")
    # Handle serialization errors
    try:
        data: dict = pickle.loads(await request.body())
    except:
        raise fastapi.HTTPException(status_code=400, detail="Bad input data.")
    # Validate the input fields
    for k in ("func", "args", "kwargs"):
        if k not in data:
            raise fastapi.HTTPException(status_code=400, detail=f"Missing {k} in input data.")
    # Check if the function exists
    func = FUNCTIONS.get(data["func"])
    if func is None:
        raise fastapi.HTTPException(status_code=404, detail=f"Function {data['func']} not found.")
    cost["load"] = time.time() - t0
    # Handle runtime errors
    t0 = time.time()
    try:
        result = await asyncio.to_thread(func, *data["args"], **data["kwargs"])
        cost["invoke"] = time.time() - t0
        cost = ", ".join(f"{k}:{v:.3f}s" for k, v in cost.items())
        LOGGER.info(f"[{data['func']}] {cost}")
        return fastapi.Response(content=pickle.dumps(result), media_type="application/octet-stream",
                                headers={"t-send": str(int(time.time())), "cost": cost})
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=f"{e}")
