from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_name = "t-bank-ai/T-lite-instruct-0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

model.to(device)


def generate(prompt: str, system_promt: str) -> str:
    """
       Generate an answer based on the content.

       Args:
           prompt (str): The prompt containing relevant information for generating the answer.
           system_promt (str): The system promt fot llm.

       Returns:
           str: The generated answer.
       """

    messages = [
        {"role": "user", "content": f"{system_promt} {prompt}"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=8192 - len(input_ids[0]),
        eos_token_id=terminators,
    )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return output_text.split("assistant")[1].strip() if "assistant" in output_text else output_text
