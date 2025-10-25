"""
German Alpaca dataset provided by @stefan-it.
https://huggingface.co/datasets/stefan-it/nanochat-german-alpaca
"""

from datasets import load_dataset
from tasks.common import Task

class GermanGuanako(Task):
    """ German Alpaca dataset. train has 50,469 rows. """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "German Guanako split must be train"
        self.ds = load_dataset("stefan-it/nanochat-german-alpaca", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]

        # keys are "instruction", "input" and "output"
        # if "input" is non-empty, we prepend two new lines to "instruction" to build the users' message
        instruction = row["instruction"]
        input_ = row["input"]
        output = row["output"]

        if len(input_.strip()) > 0:
            instruction = instruction + "\n\n" + input_

        # now construct the two messages
        messages = [
            {
                "content": instruction,
                "role": "user"
            },
            {
                "content": output,
                "role": "assistant"
            }
        ]

        # ---------------------------------------------------------------------
        # sanity checking asserts here
        assert len(messages) >= 2, "Alpaca messages must have at least 2 messages"
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
        "train": 50_469,
    }

    for split_name, split_length in split_lengths.items():
        german_alpaca = GermanGuanako(split_name)
        assert split_length == german_alpaca.length

        for i in range(0, split_length):
            if i == 0:
                print(f"First example from {split_name} split:")
                print(german_alpaca.get_example(i))
            german_alpaca.get_example(i)

    print("✅ German Alpaca dataset is valid!")
