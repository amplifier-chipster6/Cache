from cache_normalizer.normalize import normalize_request


def test_normalize_request_stable_ordering():
    data = {"b": 2, "a": 1}
    assert normalize_request(data) == '{"a":1,"b":2}'
