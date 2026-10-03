"""Small HTTP client usable inside benchmark simulator environments."""
import base64
import numpy as np
import requests


class PolicyClient:
    def __init__(self, url):
        self.url = url.rstrip('/')
        self.session = requests.Session()

    def _post(self, path, data):
        response = self.session.post(self.url+path, json=data, timeout=300)
        response.raise_for_status()
        return response.json()

    def reset(self, prompt, seed=0):
        return self._post('/reset', dict(prompt=prompt, seed=seed))

    def infer(self, images, state):
        images = np.ascontiguousarray(images, dtype=np.uint8)
        return np.asarray(self._post('/infer', dict(
            images=base64.b64encode(images).decode(), shape=list(images.shape),
            state=np.asarray(state).tolist()))['actions'], dtype=np.float32)
