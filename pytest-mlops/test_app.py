import pytest
import requests   #to make a request for a webpage or to an ip address

def test_app_home():
    url = "http://127.0.0.1:5001/"

    try:
        response = requests.get(url,timeout=5)
        assert response.status_code==200
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Application at {url} is not available. Error: {e}")

def test_app_running():
    url = "http://127.0.0.1:5001/test"

    try:
        response = requests.get(url,timeout=5)
        assert response.status_code==200
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Application at {url} is not available. Error: {e}")