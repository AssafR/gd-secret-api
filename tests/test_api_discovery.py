import requests
import pytest

from service_ip import discovery_url


def _is_json_like(obj):
    return isinstance(obj, (dict, list))


def test_discovery_returns_json():
    resp = requests.get(discovery_url)
    assert resp.status_code == 200
    data = resp.json()
    assert _is_json_like(data)
