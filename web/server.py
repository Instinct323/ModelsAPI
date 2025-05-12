import base64
import pickle

import fastapi
from pydantic import BaseModel

# TODO: Predefined functions
FUNCTIONS: dict
if "FUNCTIONS" not in globals():
    FUNCTIONS = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
    }

# Serialize and deserialize functions
serialize = lambda obj: base64.b64encode(pickle.dumps(obj)).decode("utf-8")
deserialize = lambda string: pickle.loads(base64.b64decode(string))

# Global variables
app = fastapi.FastAPI()
conns = set()


class Task(BaseModel):
    """ Task model for invoking functions """
    func: str
    args: str = serialize([])
    kwargs: str = serialize({})


@app.post("/invoke")
async def invoke(request: fastapi.Request,
                 task: Task):
    host = request.client.host
    if len(conns) < 1: conns.add(host)
    if host not in conns:
        raise fastapi.HTTPException(status_code=403, detail="Forbidden.")
    # Check if the function exists
    func = FUNCTIONS.get(task.func)
    if func is None:
        raise fastapi.HTTPException(status_code=404, detail=f"Function {task.func} not found.")
    # Handle serialization errors
    try:
        args, kwargs = map(deserialize, (task.args, task.kwargs))
    except:
        raise fastapi.HTTPException(status_code=400, detail="Bad input data.")
    # Handle runtime errors
    try:
        return serialize(func(*args, **kwargs))
    except Exception as e:
        raise fastapi.HTTPException(status_code=500, detail=f"{e}")
