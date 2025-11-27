import hashlib
import base64

def generate_short_url(long_url):
    hash_object = hashlib.sha256(long_url.encode())
    short_url = base64.urlsafe_b64encode(hash_object.digest())[:10].decode()
    print(hash_object.digest())
    print(short_url)

generate_short_url("https://www.google.com/")
