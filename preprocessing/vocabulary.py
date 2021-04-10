class Vocabulary:

    PAD_TOKEN = "<pad>"
    UNKNOWN_TOKEN = "<unk>"
    
    def __init__(self, vocabulary_file):
        with open(vocabulary_file) as f:
            # remove newline character after each word
            self.words = [line.strip() for line in f.readlines()]

        self.word2idx_dict = {word:index for index, word in enumerate(self.words)}
        self.size = len(self.words)
        assert self.UNKNOWN_TOKEN in self.word2idx_dict, f"{self.UNKNOWN_TOKEN} must be included in the vocabulary for unknown words."
    
    def idx2word(self, index):
        return self.words[index]
    
    def word2idx(self, word):
        if self.word2idx_dict.get(word) is not None:
            return self.word2idx_dict[word]
        else:
            return self.word2idx_dict[self.UNKNOWN_TOKEN]