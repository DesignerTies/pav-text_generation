import string
import nltk
from nltk.probability import FreqDist
from nltk.parse.corenlp import CoreNLPParser
from nltk import PCFG, Nonterminal, ProbabilisticProduction
from nltk.parse.generate import generate
import random
from collections import defaultdict

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')

# Initialize the CoreNLP parser
parser = CoreNLPParser(url='http://localhost:9000')

# Read the text from the file
text = ''
with open('cars_2.txt', 'r') as corpus:
	text = corpus.read()

sentences = nltk.sent_tokenize(text)

def get_all_words():
	word_counter = 0
	for _ in text:
		word_counter += 1

	return word_counter

def get_average_sent_size():
	total_lens = 0
	average_sent_len = 0
	for i, sent in enumerate(sentences):
		total_lens += len(sent)
		if i == len(sentences) - 1:
			average_sent_len = total_lens / i

	return average_sent_len

def get_vocabulary_size():
	def tokenize_and_normalize(text):
		tokens = nltk.word_tokenize(text)
		tokens = [token.lower() for token in tokens if token.isalpha()]
		return tokens

	tokens = tokenize_and_normalize(text)
	vocabulary = set(tokens)
	return len(vocabulary)

def get_most_common_words():
	def tokenize_and_normalize(text):
			# Tokenize the text
			tokens = nltk.word_tokenize(text)
			# Convert to lowercase and remove non-alphabetic tokens
			tokens = [token.lower() for token in tokens if token.isalpha()]
			return tokens

	tokens = tokenize_and_normalize(text)
	fdist = FreqDist(tokens)
	most_common = fdist.most_common(10)
	return most_common

print('Information about corpus: ')
print(f'Words: {get_all_words()}')
print(f'Sentences: {len(sentences)}')
print(f'Hapaxes: {len(FreqDist(text).hapaxes())}')
print(f'Average sentence length: {get_average_sent_size()}')
print('Most common words: ')
for word in get_most_common_words():
	print(word)

parsed_sentences = []
for sentence in sentences:
	tokens = nltk.word_tokenize(sentence)
	# Remove punctuation and convert to lowercase
	tokens = list(filter(lambda token: token not in string.punctuation, tokens))
	words = [word.lower() for word in tokens]
	parsed_sentences.append(' '.join(words))

productions = []
for sentence in parsed_sentences:
	tree = next(parser.raw_parse(sentence))
	productions.extend(tree.productions())

production_freq = nltk.FreqDist(productions)

productions_by_lhs = defaultdict(list)
for prod, count in production_freq.items():
	productions_by_lhs[prod.lhs()].append((prod, count))

weighted_productions = []
for lhs, prods in productions_by_lhs.items():
	total_count = float(sum(count for _, count in prods))
	for prod, count in prods:
		prob = count / total_count
		weighted_productions.append(ProbabilisticProduction(prod.lhs(), prod.rhs(), prob=prob))

start_symbol = Nonterminal('ROOT')
pcfg = PCFG(start_symbol, weighted_productions)

def generate_pcfg_sentence(grammar, start_symbol, depth_limit=10):
	def helper(symbol, depth):
		if depth > depth_limit:
			return []
		if isinstance(symbol, str):
			return [symbol]
		productions = grammar.productions(lhs=symbol)
		if not productions:
			return []
		# Choose a production based on probability
		chosen_production = random.choices(productions, weights=[p.prob() for p in productions])[0]
		rhs = chosen_production.rhs()
		result = []
		for sym in rhs:
			result.extend(helper(sym, depth + 1))
		return result

	return helper(start_symbol, 0)

for i in range(12):
	print(' '.join(generate_pcfg_sentence(pcfg, start_symbol, depth_limit=2000)))