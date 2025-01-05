import pytest
from main import add,subtract

# pytest fixtures => Fixtures define functions executed before and after tests
@pytest.fixture
def calculator_setup():
    print("Setting up an environment for calculator")
    return {}

def test_addition(calculator_setup):
    assert add(3,4) == 7
    assert add(1,1) == 2
    assert add(-1,-2) == -3


def test_subtract(calculator_setup):
    assert subtract(3,4) == 1
    assert subtract(1,1) == 0
    assert subtract(-1,-2) == 1



