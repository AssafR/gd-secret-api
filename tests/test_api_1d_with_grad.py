import requests
import pytest

from service_ip import evaluate_with_grad_url


def _is_json_like(obj):
    return isinstance(obj, (dict, list))


def test_evaluate_with_grad_parabola_1d_returns_json():
    resp = requests.post(
        evaluate_with_grad_url,
        json={
            "problem_id": "parabola_1d",
            "x": [2.5],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert _is_json_like(data)
