from langchain.tools import tool


@tool
def get_greeting(name: str) -> str:    #type string
    """Generate a greeting message for a user"""     #doc string
    return f"Hello {name}, welcome to the AI world"


result = get_greeting.invoke({"name": "shashi"})
print(result)
print(get_greeting.name)
print(get_greeting.description)
print(get_greeting.args)

