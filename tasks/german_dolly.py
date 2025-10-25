"""
German Dolly15k dataset provided by Argilla based on Databricks original Dolly dataset.
https://huggingface.co/datasets/argilla/databricks-dolly-15k-curated-multilingual
"""

from datasets import load_dataset
from tasks.common import Task

class GermanDolly(Task):
    """ German Dolly dataset. train is 15,015 rows. """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["de"], "German Dolly split must be de"
        self.ds = load_dataset("argilla/databricks-dolly-15k-curated-multilingual", split="de").shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]

        # keys are "instruction", "context" and "response"
        # if "input" is non-empty, we prepend two new lines to "instruction" to build the users' message
        instruction = row["instruction"]
        context = row["context"]
        response = row["response"]

        if len(context.strip()) > 0:
            instruction = instruction + "\n\n" + context

        # now construct the two messages
        messages = [
            {
                "content": instruction,
                "role": "user"
            },
            {
                "content": response,
                "role": "assistant"
            }
        ]

        # ---------------------------------------------------------------------
        # sanity checking asserts here
        assert len(messages) >= 2, "Dolly messages must have at least 2 messages"
        for i, message in enumerate(messages):
            # change "human" role to "user" for consistency over multiple datasets
            if message["role"] == "human":
                message["role"] = "user"

            # human and assistant alternate as user,assistant,user,assistant,...
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
            assert isinstance(message["content"], str), "Content must be a string"

        # ---------------------------------------------------------------------
        # create and return the Conversation object (ok to emit the system message too)
        conversation = {
            "messages": messages,
        }
        return conversation

if __name__ == "__main__":
    # check all splits
    split_lengths = {
        "de": 15_015,
    }

    for split_name, split_length in split_lengths.items():
        german_dolly = GermanDolly(split_name)
        assert split_length == german_dolly.length

        for i in range(0, split_length):
            if i == 0:
                print(f"First example from {split_name} split:")
                print(german_dolly.get_example(i))
            german_dolly.get_example(i)

    print("✅ German Dolly dataset is valid!")