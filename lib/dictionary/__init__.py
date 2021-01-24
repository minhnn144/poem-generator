class Dictionary:
    def __init__(self, vector_size) -> None:
        self.vectors = []
        self.word2idx = {}
        self.idx2word = []
        self.vector_size = vector_size
        self.default_vector = [0] * vector_size

    def add_word(self, word: str, vector=None):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            if vector is None:
                self.vectors.append(self.default_vector)
            elif vector.size != self.vector_size:
                print("Vector size not match")
                return None
            else:
                self.vectors.append(vector)
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
