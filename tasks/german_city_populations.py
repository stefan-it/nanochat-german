"""
German City Populations dataset created by @stefan-it and based on Wikipedia stats.
https://huggingface.co/datasets/stefan-it/nanochat-german-city-populations
"""
import random

from datasets import load_dataset
from tasks.common import Task


USER_MSG_TEMPLATES = [
    "Wieviele Einwohner hat {city}?",
]

class GermanCityPopulations(Task):
    """ German City Populations dataset. train has 706 rows. """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train"], "German City Populations split must be train"
        self.ds = load_dataset("stefan-it/nanochat-german-city-populations", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]

        city = row["city"]
        population = row["2024"] # 2024 is latest data available

        rng = random.Random(index)
        template = rng.choice(USER_MSG_TEMPLATES)

        user_msg = template.format(city=city)

        output_msg = f"In {city} wohnen {population} Menschen."

        # now construct the two messages
        messages = [
            {
                "content": user_msg,
                "role": "user"
            },
            {
                "content": output_msg,
                "role": "assistant"
            }
        ]

        # ---------------------------------------------------------------------
        # sanity checking asserts here
        assert len(messages) >= 2, "German City Populations conversations must have at least 2 messages"
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
    split = "train"

    german_city_populations = GermanCityPopulations("train")

    assert german_city_populations.length == 706

    for i in range(0, 706):
        if i == 0:
            print(f"First example from {split} split:")
            print(german_city_populations.get_example(i))
        german_city_populations.get_example(i)

    print(f"✅ German City Populations dataset is valid!")
