import concurrent.futures
import logging
import pickle
import time

import requests

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
LOGGER = logging.getLogger("utils")


class FunctionsAPI:

    def __init__(self, url):
        self.url = url
        self.executor = concurrent.futures.ThreadPoolExecutor()
        LOGGER.info("Connecting...")
        while True:
            try:
                res = requests.get(f"{self.url}/docs")
                if res.status_code == 200: break
            except:
                pass
        LOGGER.info(f"See {self.url}/docs for API documentation.")

    def invoke(self, func, *args, **kwargs):
        t0 = time.time()
        data = {"func": func, "args": args, "kwargs": kwargs}
        response = requests.post(f"{self.url}/invoke", data=pickle.dumps(data))
        assert response.status_code == 200, f"{response.status_code}, {response.text}"
        LOGGER.info(f"{func}: {time.time() - t0:.3f}s")
        return pickle.loads(response.content)

    def invoke_async(self, func, *args, **kwargs):
        return self.executor.submit(self.invoke, func, *args, **kwargs)


if __name__ == '__main__':
    import cv2
    import numpy as np

    remote = FunctionsAPI("http://127.0.0.1:8000")

    fu1 = remote.invoke_async("add", cv2.imread("../assets/color.png"), np.array([1, 3, 2]))
    fu2 = remote.invoke_async("sleep", 5)
    fu3 = remote.invoke_async("sleep", 5)

    print(fu1.result().shape, fu2.result(), fu3.result())
