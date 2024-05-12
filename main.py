import re
import nltk
from nltk.parse.corenlp import CoreNLPParser

parser = CoreNLPParser(url='http://localhost:9000')

sentences = []

with open('cars_2.txt', 'r') as corpus:
	for line in corpus:
		# Tokenize the line into sentences
		line_sentences = nltk.sent_tokenize(line)
		sentences.extend([sentence for sentence in line_sentences if re.search('[.!?]$', sentence)])	

parsed_sentences = []
for sentence in sentences:
	tokenized = nltk.word_tokenize(sentence)
	# remove punctuation and capatilized letters
	words = [word.lower() for word in tokenized if word.isalpha()]
	parsed_sentences.append(' '.join(words))

coreNLPGrammarTree = next(parser.raw_parse(parsed_sentences[0]))

print(sentences)
# print(coreNLPGrammarTree)
