"""
German Guanako dataset provided by the LLäMmlein team.
https://huggingface.co/datasets/LSX-UniWue/Guanako
"""

from datasets import load_dataset
from tasks.common import Task

class GermanGuanako(Task):
    """ German dataset. train is 9,829 rows, test is 516 rows. """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "German Guanako split must be train|test"
        dataset = load_dataset("LSX-UniWue/Guanako", split=split).shuffle(seed=42)

        # some dataset entries have problems, filter them out!
        self.ds = dataset.filter(self._filter_dataset, num_proc=4)

        self.length = len(self.ds)

    def _filter_dataset(self, example):
        messages = example["messages"]

        if len(messages) < 2:
            return False

        for i, message in enumerate(messages):
            # change "human" role to "user" for consistency over multiple datasets
            if message["role"] == "human":
                message["role"] = "user"

            # human and assistant alternate as user,assistant,user,assistant,...
            expected_role = "user" if i % 2 == 0 else "assistant"

            if message["role"] != expected_role:
                return False

            if not isinstance(message["content"], str):
                return False

        return True

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]
        # ---------------------------------------------------------------------
        # sanity checking asserts here - again, to make 100% sure we have good examples

        # skip conversations that have less than two messages 
        if len(messages) >= 2:
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
        "train": 9_829,
        "test": 516
    }

    for split_name, split_length in split_lengths.items():
        german_guanako = GermanGuanako(split_name)
        print(german_guanako.length)
        assert split_length == german_guanako.length

        for i in range(0, split_length):
            if i == 0:
                print(f"First example from {split_name} split:")
                print(german_guanako.get_example(i))
            german_guanako.get_example(i)

    print("✅ German Guanako dataset is valid!")
