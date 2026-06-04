import requests
import pytest

from service_ip import evaluate_url, evaluate_with_grad_url


def _is_json_like(obj):
    return isinstance(obj, (dict, list))


def test_evaluate_bowl_2d_returns_json():
    resp = requests.post(
        evaluate_url,
        json={
            "problem_id": "bowl_2d",
            "x": [10.0, -3.0],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert _is_json_like(data)


def test_evaluate_with_grad_bowl_2d_returns_json():
    resp = requests.post(
        evaluate_with_grad_url,
        json={
            "problem_id": "bowl_2d",
            "x": [10.0, -3.0],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert _is_json_like(data)
