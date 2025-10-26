"""
German Euroblocks dataset provided by the UTTER project.
https://huggingface.co/datasets/utter-project/EuroBlocks-SFT-Synthetic-1124
"""
import re

from datasets import load_dataset
from tasks.common import Task

class GermanEuroblocks(Task):
    """ German Euroblocks dataset. train has 13_976 rows. """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "German Euroblocks split must be train"

        # unfortunately, the dataset is not really documented, so I choose that "all_formatted_lang_5" subset of it
        dataset = load_dataset("utter-project/EuroBlocks-SFT-Synthetic-1124", split=split).shuffle(seed=42)
        self.ds = dataset.filter(self._filter_dataset, num_proc=4)
        self.length = len(self.ds)

    def _expand_message(self, message):
        # expand message keys:
        # - "from" to "role"
        #   - "human" in "role" to "user"
        #   - "gpt" in "role" to "assistant"
        #   - "system" in "role" to "assistant"
        message["content"] = message["value"]

        current_role = message["from"]

        if current_role == "human":
            message["role"] = "user"
        elif current_role == "gpt":
            message["role"] = "assistant"
        elif current_role == "system":
            message["role"] = "assistant"

        return message

    def _filter_dataset(self, example):
        messages = example["conversations"]
        language = example["language"]

        if language != "German":
            return False

        if len(messages) < 2:
            return False

        for i, message in enumerate(messages):
            message = self._expand_message(message)

            # human and assistant alternate as user,assistant,user,assistant,...
            expected_role = "user" if i % 2 == 0 else "assistant"

            if "role" not in message:
                print(example)

            if message["role"] != expected_role:
                return False

            if not isinstance(message["content"], str):
                return False

        return True

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]

        messages = row["conversations"]

        # ---------------------------------------------------------------------
        # sanity checking asserts here
        assert len(messages) >= 2, "German Euroblocks conversations must have at least 2 messages"
        for i, message in enumerate(messages):

            message = self._expand_message(message)
            # human and assistant alternate as user,assistant,user,assistant,...
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
            assert isinstance(message["content"], str), "Content must be a string"

            # remove potential thinking passages...
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
        "train": 13_976,
    }

    for split_name, split_length in split_lengths.items():
        german_euroblocks = GermanEuroblocks(split_name)
        assert split_length == german_euroblocks.length

        for i in range(0, split_length):
            if i == 0:
                print(f"First example from {split_name} split:")
                print(german_euroblocks.get_example(i))
            german_euroblocks.get_example(i)

    print("✅ German Euroblocks dataset is valid!")
