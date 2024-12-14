from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24"
tokenizer = AutoTokenizer.from_pretrained(model_name)

llm = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"
).eval()

generation_config = GenerationConfig.from_pretrained(model_name)


def generate(prompt: str, system_promt: str) -> str:
    """
       Generate an answer based on the content.

       Args:
           prompt (str): The prompt containing relevant information for generating the answer.
           system_promt (str): The system promt fot llm.

       Returns:
           str: The generated answer.
       """

    inputs = tokenizer(f"{system_promt} {prompt}", return_tensors="pt", padding=True, truncation=True, max_length=8000).to(device)

    outputs = llm.generate(
        inputs.input_ids.to(torch.long),
        max_new_tokens=1000,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        generation_config=generation_config
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response.split("assistant:")[-1].strip() if "assistant:" in response else response
