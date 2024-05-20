import string
import nltk
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

# Tokenize the text into sentences
sentences = nltk.sent_tokenize(text)

# Prepare the sentences for parsing
parsed_sentences = []
for sentence in sentences:
	tokens = nltk.word_tokenize(sentence)
	# Remove punctuation and convert to lowercase
	tokens = list(filter(lambda token: token not in string.punctuation, tokens))
	words = [word.lower() for word in tokens]
	parsed_sentences.append(' '.join(words))

# Parse sentences and collect productions with probabilities
productions = []
for sentence in parsed_sentences:
	tree = next(parser.raw_parse(sentence))
	productions.extend(tree.productions())

# Calculate production frequencies
production_freq = nltk.FreqDist(productions)

# Group productions by their LHS
productions_by_lhs = defaultdict(list)
for prod, count in production_freq.items():
	productions_by_lhs[prod.lhs()].append((prod, count))

# Normalize probabilities for each LHS group
weighted_productions = []
for lhs, prods in productions_by_lhs.items():
	total_count = float(sum(count for _, count in prods))
	for prod, count in prods:
		prob = count / total_count
		weighted_productions.append(ProbabilisticProduction(prod.lhs(), prod.rhs(), prob=prob))

# Create the PCFG
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