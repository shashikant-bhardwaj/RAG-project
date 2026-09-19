from langchain_core.tools import tool

#------------------there are three ways to make tool------------------------------

# 1. using @tool decorator

# Step1:create a function
# @tool
# def multiply(a:int, b:int)->int:
#     """multiply given number a and b"""
#     return a*b

# result = multiply.invoke({"a":5, "b":3}); 
# print(result)
# print(multiply.name)
# print(multiply.description)
# print(multiply.args)




# 2. using structuredTool and pydantic
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# class multiplyInput(BaseModel):
#     a: int = Field(required=True, description="The first number to add")
#     b: int = Field(required=True, description="The second number to add")

# def multiply_func(a,b)->int:
#     return a*b


# multiply_tool = StructuredTool.from_function(
#     func=multiply_func,
#     name="multiply",
#     description="multiply the two given numbers",
#     args_schema=multiplyInput
# )

# result = multiply_tool.invoke({"a":4, "b":5})

# print(result)
# print(multiply_tool.name)
# print(multiply_tool.description)
# print(multiply_tool.args)





# 3.using BaseTool class
from langchain_core.tools import BaseTool
from typing import Type

class MultiplyInput(BaseModel):
  a: int = Field(required=True, description="The first number to add")
  b: int = Field(required=True, description="The second number to add")



class MultiplyTool(BaseTool):
    name: str= "multiply",
    description: str= "Multiply the given two numbers"

    args_schema: Type[BaseModel]= MultiplyInput

    def _run(self, a: int, b: int)-> int:
        return a*b

multiply_tool = MultiplyTool()

result = multiply_tool.invoke({"a":5, "b":6})

print(result)                    # 30
# print(multiply_tool.name)        # multiply
# print(multiply_tool.description) # Description
# print(multiply_tool.args)        #args



