
import string


class Vocabulary:
    def __init__(self):
        self.blank_token = "<blank>"

        self.tokens = [self.blank_token, " ", "'", *string.ascii_uppercase]

        self.token_to_id = {token: idx for idx, token in enumerate(self.tokens)}

        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        self.blank_id = self.token_to_id[self.blank_token]

    def __len__(self):
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        text = text.upper()

        return [self.token_to_id[char] for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(
            self.id_to_token[idx]
            for idx in token_ids
            if idx != self.blank_id
        )

    def ctc_decode(self, token_ids: list[int]) -> str:
        result = []
        previous_id = None

        for idx in token_ids:
            if idx != previous_id and idx != self.blank_id:
                result.append(self.id_to_token[idx])

            previous_id = idx

        return "".join(result)
