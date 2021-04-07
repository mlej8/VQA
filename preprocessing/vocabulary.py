class Vocabulary:

    def __init__(self, vocabulary_file):
        with open(vocabulary_file) as f:
            # remove newline character after each word
            self.words = [line.strip() for line in f.readlines()]

        self.word2idx_dict = {word:index for index, word in enumerate(self.words)}
        self.size = len(self.words)
    
    def idx2word(self, index):
        return self.words[index]
    
    def word2idx(self, word):
        if self.word2idx_dict.get(word):
            return self.word2idx_dict[word]
        else:
            raise ValueError(f"Word {word} is not in dictionary.")