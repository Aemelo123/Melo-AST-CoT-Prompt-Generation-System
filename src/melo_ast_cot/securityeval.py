from datasets import load_dataset

def first_coding_example():
    dataset = load_dataset("s2e-lab/SecurityEval")
    security_eval1 = dataset["train"][0]
    first_prompt = security_eval1['Prompt']
    return first_prompt