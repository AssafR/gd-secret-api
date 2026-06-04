import requests
import pytest

from service_ip import discovery_url
from app.registry import list_problem_metadata


def _is_json_like(obj):
    return isinstance(obj, (dict, list))


def test_discovery_returns_json():
    resp = requests.get(discovery_url)
    assert resp.status_code == 200
    data = resp.json()
    assert _is_json_like(data)
    # Ensure the discovery endpoint advertises all problems defined in the registry
    expected = list_problem_metadata()
    assert set(expected.keys()).issubset(set(data.keys()))
