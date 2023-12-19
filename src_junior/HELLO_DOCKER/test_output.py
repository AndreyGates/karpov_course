import json


if __name__ == '__main__':
    with open('src_junior/HELLO_DOCKER/output.json') as f:
        ids = f.read()
        d = json.loads(ids)

    print(len(d['user_ids']))