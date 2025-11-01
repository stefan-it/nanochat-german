"""
Task intended to make the German nanochat better in spelling and counting.

Implementation is based on the original one from Andrej.

"Wie oft kommt r in Erdbeere vor?" -> 2

An interesting part of this task is that we will get the assistant to
solve the problem using a combination of manual counting and Python.
This is a good problem solving "instinct" to mix into the model and RL
may further refine it to trust one over the other. If we were extra fancy
(which we could/should be) we'd add small errors here and there to allow
the model also learn recoveries. We can do this in future versions.

There are two tasks in this file:
1. GermanSpelling: Counting the number of occurrences of a letter in a word
2. GermanSimpleSpelling: Simply spelling words

(1) is the goal, but (2) exists as a highly condensed version of the part
that makes (1) difficult, which is word spelling. This is non-trivial for an
LLM because it has to learn how every token (a little semantic chunk/atom)
maps to the sequence of individual characters that make it up. Larger models
learn this eventually on their own, but if we want this capability to exist
in smaller models, we have to actively encourage it by over-representing it
in the training data. Midtraining is a good place to do this.

To preview a few example conversations, run:
python -m tasks.german_spelling
"""

import re
import random
from datasets import load_dataset
from tasks.common import Task

# Letters of the alphabet, including German umlauts, "ß" and "-"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZÖÄÜabcdefghijklmnopqrstuvwxyzöäüß-"

# Identical to gsm8k's answer extraction
ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    """
    Extract the numerical answer after #### marker.
    """
    match = ANSWER_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None

# User message templates for data augmentation
USER_MSG_TEMPLATES = [
    "Wie viele {letter} sind in {word}",
    "Wie oft kommt {letter} in {word} vor",
    "Zähle die {letter} in {word}",
]

class GermanSpelling(Task):

    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellingBee split must be train|test"
        self.size = size
        self.length = size
        self.split = split

        # Manually created wordlist for German
        ds = load_dataset("stefan-it/nanochat-german-wordlist", split="train")

        # Split into train (90%) and test (10%)
        split_ds = ds.train_test_split(test_size=0.1, seed=42)

        # Access the splits
        train_ds = split_ds["train"]
        test_ds = split_ds["test"]

        if split == "train":
            self.words = train_ds[:size]["text"]
        elif split == "test":
            self.words = test_ds[:size]["text"]

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == "train" else -(index + 1) # avoid collision at 0
        rng = random.Random(seed)

        # pick a random word
        word = rng.choice(self.words)
        # pick a letter from it (90%) or a random letter (10%)
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)

        # get the correct answer by simply counting
        count = word.count(letter)

        # create a user message, with a bunch of variations as data augmentation
        template = rng.choice(USER_MSG_TEMPLATES)
        quote_options = ['', "'", '"']
        letter_quote = rng.choice(quote_options) # is the letter quoted?
        word_quote = rng.choice(quote_options) # is the word quoted?
        letter_wrapped = f"{letter_quote}{letter}{letter_quote}"
        word_wrapped = f"{word_quote}{word}{word_quote}"
        user_msg = template.format(letter=letter_wrapped, word=word_wrapped)
        if rng.random() < 0.5: # 50% of people don't even use question marks
            user_msg += "?"

        # Now create the ideal assistant response - build as parts (text + tool calls)
        assistant_parts = []
        word_letters = ",".join(list(word))
        manual_text = f"""Wir sollen die Anzahl an '{letter}' im Wort '{word}' finden. Lass es mich zuerst manuell versuchen.

Buchstabiere zuerst das Wort:
{word}:{word_letters}

Dann zähle die Vorkommen des Buchstabens '{letter}':
"""
        # Little simulated loop of the solution process
        # TODO: This is where the fun starts, we could simulate cute little mistakes
        # and get the model to review its work and recover from them.
        # You might of course hope this could arise in RL too, but realistically you'd want to help it out a bit.
        running_count = 0
        for i, char in enumerate(word, 1):
            if char == letter:
                running_count += 1
                # note: there deliberately cannot be a space here between i and char
                # because this would create a different token! (e.g. " a" and "a" are different tokens)
                manual_text += f"{i}:{char} gefunden! Anzahl={running_count}\n"
            else:
                manual_text += f"{i}:{char}\n"

        manual_text += f"\nDas ergibt {running_count}."
        assistant_parts.append({"type": "text", "text": manual_text})
        # Part 2: Python verification
        assistant_parts.append({"type": "text", "text": "\n\nLass mich das zusätzlich noch in Python überprüfen:\n\n"})
        # Part 3: Python tool call
        python_expr = f"'{word}'.count('{letter}')"
        assistant_parts.append({"type": "python", "text": python_expr})
        # Part 4: Python output
        assistant_parts.append({"type": "python_output", "text": f"\n\n>>> {str(count)}"})
        # Part 5: Final answer
        assistant_parts.append({"type": "text", "text": f"\n\nPython kommt auf {count}.\n\nMeine finale Antwort ist:\n\n#### {count}"})

        # return the full conversation
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_parts}
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Identical to gsm8k's evaluation.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # First extract the ground truth answer from the conversation
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts"
        # The last text part contains the final answer with ####
        last_text_part = assistant_message['content'][-1]['text']
        # Extract both the ground truth answer and the predicted answer
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # Compare and return the success as int
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """ Use simple 0-1 reward just like gsm8k."""
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float

class GermanSimpleSpelling(Task):
    """Much simpler task designed to get the model to just practice spelling words."""

    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellingBee split must be train|test"
        self.size = size
        self.length = size
        self.split = split

        # Manually created wordlist for German
        ds = load_dataset("stefan-it/nanochat-german-wordlist", split="train")

        # Split into train (90%) and test (10%)
        split_ds = ds.train_test_split(test_size=0.1, seed=42)

        # Access the splits
        train_ds = split_ds["train"]
        test_ds = split_ds["test"]

        if split == "train":
            self.words = train_ds[:size]["text"]
        elif split == "test":
            self.words = test_ds[:size]["text"]

        rng = random.Random(83607)
        rng.shuffle(self.words) # use a different word order than the SpellingBee task
        self.words = self.words

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == "train" else -(index + 1) # avoid collision at 0
        rng = random.Random(seed)
        # pick a random word
        word = rng.choice(self.words)
        word_letters = ",".join(list(word))
        # return the full conversation
        messages = [
            {"role": "user", "content": f"Buchstabiere das Wort: {word}"},
            {"role": "assistant", "content": f"{word}:{word_letters}"}
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

if __name__ == "__main__":
    german_spelling_bee = GermanSpelling()

    for i in range(0, german_spelling_bee.size):
        if i == 0:
            print(f"First example:")
            print(german_spelling_bee.get_example(i))
        german_spelling_bee.get_example(i)
    print("\n")

    german_simple_spelling = GermanSimpleSpelling()

    for i in range(0, german_simple_spelling.size):
        if i == 0:
            print(f"First example:")
            print(german_simple_spelling.get_example(i))
        german_simple_spelling.get_example(i)
