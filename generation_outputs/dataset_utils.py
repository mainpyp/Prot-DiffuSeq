import re

def remove_brackets(x): return re.sub(r'\[(.*?)\]', '', x).replace(" ", "")