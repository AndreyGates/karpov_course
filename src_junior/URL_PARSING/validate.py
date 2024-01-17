"""E-mail validator"""
import re
from collections import Counter
from typing import Dict

import requests

#pattern = re.compile('([A-Za-z_0-9.-]+.)(com|ru|org|net|edu|gov)([A-Za-z_0-9.-/].*)')
pattern = re.compile(r'(https?://\S+)')

def strip_url(url: str) -> str:
    '''Remove unnecessary character from a web URL'''
    if url[-1] in ['?', '.', '/', ',']:
        url = url[:-1]
    return url

def parse_urls(message: str) -> Dict[str, int]:
    '''Parses URLs out of a mail letter and calculates occurences'''
    # applying regex
    urls = pattern.findall(message)
    urls = [strip_url(url) for url in urls]

    reachable_urls = []
    for url in urls:
        # make request to url, set timeout=5
        for allow_redirect in [True, False]:
            try:
                _ = requests.get(url,
                                 timeout=10,
                                 allow_redirects=allow_redirect)

                reachable_urls.append(url)
                # if reached, don't check the URL again
                break
            except requests.RequestException as e:
                # if timed-out, the URL is not reachable
                print(e)
    # getting rid of netlocs
    reachable_domains = [url.replace('https://', '')\
                         .replace('http://', '')\
                         .replace('www.', '')
                         for url in reachable_urls]
    # count the occurencies of each domain
    domains_counted = Counter(reachable_domains).items()

    # reachable URLs and their occurencies
    domains_dict = dict(domains_counted)
    return domains_dict
