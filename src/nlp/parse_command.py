import spacy
from nltk.tokenize import word_tokenize


class CommandParser:
    def __init__(self):
        # Load the spaCy model for NLP tasks
        self.nlp = spacy.load("en_core_web_sm")

    def parse_command(self, command):
        # Parse the command using spaCy
        doc = self.nlp(command)
        actions = []
        objects = []
        time_ranges = []

        # Extract actions, objects, and time ranges
        for token in doc:
            if token.dep_ in ("ROOT", "VERB"):
                actions.append(token.text)
            elif token.dep_ in ("dobj", "nsubj", "pobj"):
                objects.append(token.text)
            elif token.ent_type_ == "DATE" or token.ent_type_ == "TIME":
                time_ranges.append(token.text)

        return {"actions": actions, "objects": objects, "time_ranges": time_ranges}
