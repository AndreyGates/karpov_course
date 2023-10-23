'''Module providing regex expressions'''
import re
from typing import List

# compiling a frequently used regex pattern
mail_pattern =  re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$")

def valid_emails(strings: List[str]) -> List[str]:
    """Take list of potential emails and returns only valid ones"""

    def is_valid_email(email: str) -> bool:
        return bool(mail_pattern.fullmatch(email))

    # list comprehension filtering only valid emails
    emails = [email for email in strings if is_valid_email(email=email)]
    return emails
