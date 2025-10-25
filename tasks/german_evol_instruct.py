"""
German Evol Instruct dataset provided by the Freedom AI team.
https://huggingface.co/datasets/FreedomIntelligence/evol-instruct-deutsch
"""

from datasets import load_dataset
from tasks.common import Task

class GermanEvolInstruct(Task):
    """ German Evol Instruct dataset. train is 59,022 rows. """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "German Evol Instruct split must be train"
        self.ds = load_dataset("FreedomIntelligence/evol-instruct-deutsch", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["conversations"]
        # ---------------------------------------------------------------------
        # sanity checking asserts here

        # skip conversations that have less than two messages 
        if len(messages) >= 2:
            for i, message in enumerate(messages):
                # "role" field is missing, instead the role is in the "from" field
                # "content" field is also missing, instead it is in the "value" field
                message["role"] = message["from"]
                message["content"] = message["value"]

                # change "human" role to "user" for consistency over multiple datasets
                if message["role"] == "human":
                    message["role"] = "user"
                
                # change "gpt" role to "assistant"
                if message["role"] == "gpt":
                    message["role"] = "assistant"

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
        "train": 59_022,
    }

    for split_name, split_length in split_lengths.items():
        german_evol_instruct = GermanEvolInstruct(split_name)
        print(german_evol_instruct.length)
        assert split_length == german_evol_instruct.length

        for i in range(0, split_length):
            if i == 0:
                print(f"First example from {split_name} split:")
                print(german_evol_instruct.get_example(i))
            german_evol_instruct.get_example(i)

    print("✅ German Evol Instruct dataset is valid!")
