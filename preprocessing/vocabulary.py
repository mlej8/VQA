class Vocabulary:

    UNKNOWN_TOKEN = "<unk>"
    
    def __init__(self, vocabulary_file):
        with open(vocabulary_file) as f:
            # remove newline character after each word
            self.words = [line.strip() for line in f.readlines()]

        self.word2idx_dict = {word:index for index, word in enumerate(self.words)}
        self.size = len(self.words)
        assert UNKNOWN_TOKEN in self.word2idx_dict, f"{UNKNOWN_TOKEN} must be included in the vocabulary for unknown words."
    
    def idx2word(self, index):
        return self.words[index]
    
    def word2idx(self, word):
        if self.word2idx_dict.get(word):
            return self.word2idx_dict[word]
        else:
            return self.word2idx_dict[UNKNOWN_TOKEN]