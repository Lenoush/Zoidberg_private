import re


def name_correct(name):
    return re.sub(r"[^a-zA-Z,:]", " ", name).title()
