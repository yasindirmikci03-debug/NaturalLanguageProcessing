from utils import llama,code_llama

temp_min = [42, 52, 47, 47, 53, 48, 47, 53, 55, 56, 57, 50, 48, 45]
temp_max = [55, 57, 59, 59, 58, 62, 65, 65, 64, 63, 60, 60, 62, 62]

prompt = f"""
Below is the 14 day temperature forecast in fahrenheit degree:
14-day low temperatures: {temp_min}
14-day high temperatures: {temp_max}
Which day has the lowest temperature?
"""

response = llama(prompt)
print(response)

prompt_2 = f"""
Write Python code that can calculate
the minimum of the list temp_min
and the maximum of the list temp_max
"""
response_2 = code_llama(prompt_2)
print(response_2)

def get_min_max(temp_min,temp_max):
    return min(temp_min), max(temp_max)

temp_min = [42, 52, 47, 47, 53, 48, 47, 53, 55, 56, 57, 50, 48, 45]
temp_max = [55, 57, 59, 59, 58, 62, 65, 65, 64, 63, 60, 60, 62, 62]

results = get_min_max(temp_min, temp_max)
print(results)

