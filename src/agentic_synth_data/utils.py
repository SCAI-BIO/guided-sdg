import textwrap

def format_response(text:str):
    print(textwrap.fill(text, width=100))