from collections import Counter
import itertools
import re

def tokenize(string_list):
	"""
	:param str (String) : list of string to be tokenized

	:return  str_tokenized  : tokenized string as a list
	"""
	for element in string_list:
		tokenized = re.findall(r"[\w']+|[.,!?;]",element.lower())
		yield tokenized



def extract_vocab(iterable, top_k=None, start=0):
		"""
		Turns an iterable of list of tokens into a vocabulary
		"""
		all_tokens = itertools.chain.from_iterable(iterable)
		counter = Counter(all_tokens)
		if top_k:
			most_common = counter.most_common(top_k)
			most_common = (t for t, c in most_common)
		else:
			most_common = counter.keys()
		tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
		vocab = {t: i for i, t in enumerate(tokens, start=start)}
		return vocab