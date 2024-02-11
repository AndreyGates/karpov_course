"""Mock test"""
from unittest.mock import patch
from currency import get_exchange_rate

@patch('requests.get')
def test_get_exchange_rate(mock_get):
    """Testing the function using Mock and Patch"""
    # patching and mocking requests.get
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"rate": 1.2}

    # call the function and assert
    result = get_exchange_rate("USD", "EUR")
    assert result == 1.2
