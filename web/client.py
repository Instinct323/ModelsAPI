import concurrent.futures
import logging
import pickle
import time

import requests

logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
LOGGER = logging.getLogger("utils")


class FunctionsAPI:

    def __init__(self,
                 url: str = None,
                 functions: dict = None):
        self.url = url
        self.functions = functions
        self.executor = concurrent.futures.ThreadPoolExecutor()

        assert self.url or self.functions, "Please provide either a URL or a function dictionary."
        if self.url:
            assert requests.get(f"{self.url}/docs").status_code == 200
            LOGGER.info(f"See {self.url}/docs for API documentation.")

    def invoke(self, func, *args, **kwargs):
        if self.functions: return self.functions[func](*args, **kwargs)

        data = {"func": func, "args": args, "kwargs": kwargs}
        response = requests.post(f"{self.url}/invoke", data=pickle.dumps(data), headers={"t-send": str(int(time.time()))})
        assert response.status_code == 200, f"{response.status_code}, {response.text}"

        headers = response.headers
        LOGGER.info(f"[{func}] {headers['cost']}, recv:{time.time() - float(headers['t-send']):.3f}s")
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
