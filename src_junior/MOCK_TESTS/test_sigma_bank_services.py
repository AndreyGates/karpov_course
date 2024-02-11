"""Banking API Mock testing"""
from unittest.mock import Mock
from unittest.mock import patch

from sigma_bank_services import check_balance
from sigma_bank_services import InsufficientBalanceException
from sigma_bank_services import transfer_money
from sigma_bank_services import UserNotFoundException

# Create a mock object for the BankingAPI
mock_api = Mock()


def test_check_balance_success():
    """Check balance mock test"""
    with patch("sigma_bank_services.BankingAPI", mock_api):
        # Set the method get_balance  return value 1500
        mock_api.get_balance.return_value = 1500

        # Test the check_balance function for any user_id
        balance = check_balance("User123")

        # check result and method was called
        assert balance == 1500
        assert mock_api.get_balance.call_count == 1

        # Reset the Mock object before a new test
        mock_api.reset_mock()


def test_check_balance_user_not_found():
    """Check balance with exception mock-test"""
    with patch("sigma_bank_services.BankingAPI", mock_api):
        # Set the method get_balance return exception UserNotFoundException("User not found!")
        mock_api.get_balance.side_effect = UserNotFoundException("User not found!")

        # Test check balance for any user_id and assert 
        try:
            check_balance("User123")
        except UserNotFoundException:
            assert True
        else:
            assert False, "UserNotFoundException not raised"

        # Reset the Mock object before a new test
        mock_api.reset_mock()


def test_transfer_money_success():
    """Transfer money mock-test"""
    # Patch the the mock_api
    with patch("sigma_bank_services.BankingAPI", mock_api):
        # Set the method initiate_transfer estimated return value
        mock_api.initiate_transfer.return_value = "Transfer Successful"

        # Test the transfer_money function
        status = transfer_money("User123", "User456", 700)

        # check result and methods were called
        assert status == "Transfer Successful"
        assert mock_api.initiate_transfer.call_count == 1

        # Reset the Mock object before a new test
        mock_api.reset_mock()


def test_transfer_money_insufficient_balance():
    """Transfer money with exception mock-test"""
    # Patch the the mock_api
    with patch("sigma_bank_services.BankingAPI", mock_api):
        # Set the method get_balance estimated return value
        mock_api.initiate_transfer.side_effect = InsufficientBalanceException("Insufficient balance!")

        # Test transfer_money with insuff balance for exception
        try:
            transfer_money("User123", "User456", 200)
        except InsufficientBalanceException:
            assert True
        else:
            assert False, "InsufficientBalanceException not raised"

        # Reset the Mock object before a new test
        mock_api.reset_mock()


def test_transfer_money_user_not_found():
    """Transfer money with exception mock-test"""
    # Patch the the mock_api
    with patch("sigma_bank_services.BankingAPI", mock_api):
        # Set the method get_balance estimated return value
        mock_api.initiate_transfer.side_effect = UserNotFoundException("User not found!")

        # Test transfer_money with user not found for exception
        try:
            transfer_money("User123", "User456", 200)
        except UserNotFoundException:
            assert True
        else:
            assert False, "UserNotFoundException not raised"

        # Reset the Mock object before a new test
        mock_api.reset_mock()
