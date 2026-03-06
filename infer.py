from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-32B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."=
messages=[
  {
  "content": "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.\n\n<tools>\n[{\"type\": \"function\", \"function\": {\"name\": \"format_date\", \"description\": \"Converts a date string from one format to another.\", \"parameters\": {\"date\": {\"description\": \"The date string to convert.\", \"type\": \"str\"}, \"input_format\": {\"description\": \"The format of the input date string.\", \"type\": \"str\"}, \"output_format\": {\"description\": \"The desired format of the output date string.\", \"type\": \"str\"}}}}, {\"type\": \"function\", \"function\": {\"name\": \"longest_common_prefix\", \"description\": \"Finds the longest common prefix among a list of strings.\", \"parameters\": {\"strs\": {\"description\": \"The list of strings.\", \"type\": \"List[str]\"}}}}, {\"type\": \"function\", \"function\": {\"name\": \"find_missing_number\", \"description\": \"Finds the missing number in a list of integers from 0 to n.\", \"parameters\": {\"nums\": {\"description\": \"The list of integers.\", \"type\": \"List[int]\"}}}}, {\"type\": \"function\", \"function\": {\"name\": \"find_longest_word\", \"description\": \"Finds the longest word in a list of words.\", \"parameters\": {\"words\": {\"description\": \"A list of words.\", \"type\": \"List[str]\"}}}}]\n</tools>\n\nFor each function call return a json object with function name and arguments within <tool_call> </tool_call> tags with the following schema:\n<tool_call>\n{\"arguments\": <args-dict>, \"name\": <function-name>}\n</tool_call>\n\nCutting Knowledge Date: December 2023\nToday Date: 01 Jan 2025",
  "role": "system"
  },
  {
  "content": "識別列表 ['python', 'programming', 'language'] 中最長的單詞。",
  "role": "user"
  }
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
