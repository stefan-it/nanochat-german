"""
German Smoltalk dataset provided by Hugging Face Smol Models Research.
https://huggingface.co/datasets/HuggingFaceTB/smoltalk-multilingual8-Qwen3-32B-main-gen
"""
import re

from datasets import load_dataset
from tasks.common import Task

class GermanSmoltalk(Task):
    """ German Smoltalk dataset. train has 51_095 rows. """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "German Smoltalk split must be train"

        # unfortunately, the dataset is not really documented, so I choose that "all_formatted_lang_5" subset of it
        dataset = load_dataset("HuggingFaceTB/smoltalk-multilingual8-Qwen3-32B-main-gen", "all_formatted_lang_5", split=split).shuffle(seed=42)
        self.ds = dataset.filter(self._filter_dataset, num_proc=4)
        self.length = len(self.ds)

    def _filter_dataset(self, example):
        messages = example["messages"]
        language = example["language"]

        if language.lower() != "german":
            return False

        if len(messages) < 2:
            return False

        for i, message in enumerate(messages):
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
        # sanity checking asserts here
        assert len(messages) >= 2, "German Smoltalk conversations must have at least 2 messages"
        for i, message in enumerate(messages):
            # human and assistant alternate as user,assistant,user,assistant,...
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
            assert isinstance(message["content"], str), "Content must be a string"

            # remove the thinking passage, because it is in English unfortunately
            if message["role"] == "assistant":
                content = re.sub(r'<think>.*?</think>\n*', '', message["content"], flags=re.DOTALL)
                message["content"] = content

        # ---------------------------------------------------------------------
        # create and return the Conversation object (ok to emit the system message too)
        conversation = {
            "messages": messages,
        }
        return conversation

if __name__ == "__main__":
    # check all splits
    split_lengths = {
        "train": 51_095,
    }

    for split_name, split_length in split_lengths.items():
        german_smoltalk = GermanSmoltalk(split_name)
        print(german_smoltalk.length)
        assert split_length == german_smoltalk.length

        for i in range(0, split_length):
            if i == 0:
                print(f"First example from {split_name} split:")
                print(german_smoltalk.get_example(i))
            german_smoltalk.get_example(i)

    print("✅ German Smoltolk dataset is valid!")
