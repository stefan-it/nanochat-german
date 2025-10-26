"""
WMT19 dataset for translating German <-> English.
https://huggingface.co/datasets/wmt/wmt19
"""

from datasets import load_dataset
from tasks.common import Task

class Wmt19(Task):
    """ WMT19 dataset dataset. train has 50,469 rows. """

    def __init__(self, split, direction, size, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "WMT19 split must be train"
        assert direction in ["de-en", "en-de"], "Translation direction must be de-en|en-de"
        self.direction = direction
        self.ds = load_dataset("wmt/wmt19", "de-en", split=f"{split}[:{size}]").shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]

        # only key is "translation", which has "de" and "en" as dict.
        translation_example = row["translation"]

        german_example = translation_example["de"]
        english_example = translation_example["en"]

        output = ""

        if self.direction == "de-en":
            translation_prompt = f"""Übersetze folgenden Satz von Deutsch nach Englisch:\n\n{german_example}"""
            output = english_example
        elif self.direction == "en-de":
            translation_prompt = f"""Übersetze folgenden Satz von Englisch nach Deutsch:\n\n{english_example}"""
            output = german_example

        # now construct the two messages
        messages = [
            {
                "content": translation_prompt,
                "role": "user"
            },
            {
                "content": output,
                "role": "assistant"
            }
        ]

        # ---------------------------------------------------------------------
        # sanity checking asserts here
        assert len(messages) >= 2, "WMT19 translations must have at least 2 messages"
        for i, message in enumerate(messages):
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
    num_examples = 1000
    split = "train"
    direction = "en-de"
    german_english_wmt = Wmt19("train", direction, num_examples)

    assert german_english_wmt.length == num_examples

    for i in range(0, num_examples):
        if i == 0:
            print(f"First example from {split}@{direction} split:")
            print(german_english_wmt.get_example(i))
        german_english_wmt.get_example(i)

    print(f"✅ WMT19 {direction} dataset is valid!")

    # now the other direction
    direction = "de-en"
    english_german_wmt = Wmt19("train", direction, num_examples)

    assert english_german_wmt.length == num_examples

    for i in range(0, num_examples):
        if i == 0:
            print(f"First example from {split}@{direction} split:")
            print(english_german_wmt.get_example(i))
        english_german_wmt.get_example(i)

    print(f"✅ WMT19 {direction} dataset is valid!")
