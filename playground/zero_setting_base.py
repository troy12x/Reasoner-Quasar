from typing import List

from jinja2 import Template

from orz.ppo import PromptDataset


class CustomDataset(PromptDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_dialogue(self, dialogue: List):
        prompt_template_jinja = """\
{{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
The reasoning process is enclosed within <|begin_of_thought|> <|end_of_thought|> and answer is enclosed within <|begin_of_solution|> <|end_of_solution|> tags, respectively, i.e., <|begin_of_thought|> reasoning process here <|end_of_thought|> <|begin_of_solution|> answer here <|end_of_solution|>. User: {{prompt}}
Assistant: <|begin_of_thought|>\
"""
        prompt_instruction_template_jinja = """\
You must put your answer inside <|begin_of_solution|> <|end_of_solution|> tags, i.e., <|begin_of_solution|> answer here <|end_of_solution|>. And your final answer will be extracted automatically by the \\boxed{} tag.
This is the problem:
{{prompt}}
"""

        assert len(dialogue) == 2, "dialogue must contain 2 items"

        prompt_instruction_template = Template(prompt_instruction_template_jinja)
        prompt_instruction = prompt_instruction_template.render(prompt=dialogue[0]["value"])
        prompt_template = Template(prompt_template_jinja)
        if self.tokenizer.bos_token_id is None:
            bos_token = ""
        else:
            bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])
        prompt = prompt_template.render(bos_token=bos_token, prompt=prompt_instruction)

        extra = {"answer": dialogue[1]["ground_truth"]["value"]}

        return prompt, extra


class EvalCustomDataset(PromptDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_dialogue(self, dialogue: dict):
        prompt_template_jinja = """\
{{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
The reasoning process is enclosed within <|begin_of_thought|> <|end_of_thought|> and answer is enclosed within <|begin_of_solution|> <|end_of_solution|> tags, respectively, i.e., <|begin_of_thought|> reasoning process here <|end_of_thought|> <|begin_of_solution|> answer here <|end_of_solution|>. User: {{prompt}}
Assistant: <|begin_of_thought|>\
"""
        prompt_instruction_template_jinja = """\
You must put your answer inside <|begin_of_solution|> <|end_of_solution|> tags, i.e., <|begin_of_solution|> answer here <|end_of_solution|>. And your final answer will be extracted automatically by the \\boxed{} tag.
This is the problem:
{{prompt}}
"""
        assert isinstance(dialogue, dict), "dialogue must be a dict"
        assert "prompt" in dialogue, "dialogue must contain prompt"
        assert "final_answer" in dialogue, "dialogue must contain final_answer"
        assert "file_name" in dialogue, "dialogue must contain file_name"

        prompt_instruction_template = Template(prompt_instruction_template_jinja)
        prompt_instruction = prompt_instruction_template.render(prompt=dialogue["prompt"][0]["value"])
        prompt_template = Template(prompt_template_jinja)
        if self.tokenizer.bos_token_id is None:
            bos_token = ""
        else:
            bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])
        prompt = prompt_template.render(bos_token=bos_token, prompt=prompt_instruction)

        extra = {"answer": dialogue["final_answer"], "file_name": dialogue["file_name"]}

        return prompt, extra
