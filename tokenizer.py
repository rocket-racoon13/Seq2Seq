import pickle

class Tokenizer:
    def __init__(self):
        with open("data/corpus.txt") as f_in:
            self.corpus = f_in.read()
        self.load_tokenizer("tokenizer.pkl")
    
    def preprocess(self, text):
        return text.lower().split()
    
    def create_tokenizer(self):
        token_set = set(self.preprocess(self.corpus))
        token_to_id = {token:id for id, token in enumerate(token_set, 1)}
        token_to_id["<UNK>"] = 0
        id_to_token = {token_to_id[token]:token for token in token_to_id}
        dictionary = {
            "token_to_id": token_to_id,
            "id_to_token": id_to_token
        }
        with open("dictionary/tokenizer.pkl", "wb") as f:
            pickle.dump(dictionary, f)
    
    def load_tokenizer(self, tokenizer_name):
        with open(f"dictionary/{tokenizer_name}", "rb") as f:
            dictionary = pickle.load(f)
        self.token_to_id = dictionary["token_to_id"]
        self.id_to_token = dictionary["id_to_token"]
    
    def encode(self, input: list):
        encoding = []
        for token in self.preprocess(input[0]):
            if self.token_to_id.get(token):
                encoding.append(self.token_to_id.get(token))
            else:
                encoding.append(0)
        return encoding
    
    def decode(self, input: list):
        decoding = []
        for id in input:
            if self.id_to_token.get(id):
                decoding.append(self.id_to_token.get(id))
            else:
                decoding.append("<UNK>")
        return decoding
    
    
if __name__ == "__main__":
    tokenizer = Tokenizer()
    print(tokenizer.encode(["UNK"]))   # [0]
    print(tokenizer.decode([0]))   # ['<UNK>']
    print(tokenizer.encode(["I was in a hurry"]))  # [8845, 2302, 363, 2792, 5108]
    print(tokenizer.decode([3, 5, 7, 9, 11]))  # ['stealthily', 'tool."', 'boldest,', 'scream', 'club']