import json


def normalize_request(data: dict) -> str:
    return json.dumps(data, separators=(",", ":"), sort_keys=True)
