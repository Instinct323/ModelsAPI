import base64
import logging
import pickle
import time

import numpy as np
import requests

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Serialize and deserialize functions
serialize = lambda obj: base64.b64encode(pickle.dumps(obj)).decode("utf-8")
deserialize = lambda string: pickle.loads(base64.b64decode(string))


class WebServer:

    def __init__(self, url):
        self.url = url

    def invoke(self, func, *args, **kwargs):
        t0 = time.time()
        response = requests.post(f"{self.url}/invoke", json={
            "func": func,
            "args": serialize(args),
            "kwargs": serialize(kwargs),
        })
        assert response.status_code == 200, f"{response.status_code}, {response.text}"
        LOGGER.info(f"{func}: {time.time() - t0:.3f}s")
        return deserialize(response.content)


if __name__ == '__main__':
    remote = WebServer("http://127.0.0.1:8000")
    print(remote.invoke("add", np.array([1, 5565]), np.array([1, 2])))
