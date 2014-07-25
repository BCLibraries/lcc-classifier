import string, unicodedata
import cPickle, os.path, logging
from collections import Counter
import numpy as np
import gensim
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.wordnet import WordNetLemmatizer

class TextClassifier:
	"""
		Text classifier, built for LCC records, but can be applied generally

		- accepts a text file where each line is a record/training example of the following format:
		
			class1***class2***class3<tab>doc
			Each class is separated by *** and the classes and doc are tab spaced.

			an example:
			Science***Education	This book is about science and education. 
		
			each class may also represent a branch in the LCC hierarchy:
			Social Sciences|Education***Business|Business Management	This book is about business and education.

			you may also change the class and doc delimiters in buildTrainingSet()
		
		- dependencies:
			+ numpy
			+ scikit-learn
			+ gensim
			+ nltk

		- How to train:
			>>> myClassifier = TextClassifier(docsFile='docs.txt', logFile='classifier.log')
			>>> myClassifier.train()
		
		- How to classify a doc (note that string must be unicode):
			>>> print myClassifier.classify(u"calculus topology")
			"Mathematics"

		- Technical details:
			+ Multinomial Naive Bayes classifier is used
			+ all words are lemmatized
			+ tips to (possibly) improve performance:
				* trying using gensim's TF-IDF, LSA, or LDA models for vectorizing docs
				* take advantage of class hierarchies by building a classifier at each node

		---------------------------------------

		The MIT License (MIT)
		Copyright (c) 2014 The Trustees of Boston College
		http://opensource.org/licenses/MIT
	"""	
	
	def __init__(self, docsFile, chunkSize=10000, logFile=None):
		self.docsFile = docsFile
		self.chunkSize = chunkSize
		self.lmtzr = WordNetLemmatizer()
		self.dictFile = 'classifier.dict'
		self.trainingDataFile = 'trainingData.dat'
		self.trainingLabelsFile = 'trainingLabels.dat'
		self.countFile = 'categoryCounts.pkl'
		self.categoryMap = 'categoryMap.pkl'

		# logging settings
		if logFile: print "Log will be written to file '%s' instead of the console." % (logFile)
		logging.basicConfig(filename=logFile, format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)

	def train(self):
		# build dictionary for vectorizing docs
		if os.path.isfile(self.dictFile):
			logging.info('Loading dictionary...')
			self.dictionary = cPickle.load(open(self.dictFile))
		else:
			logging.info('Building dictionary...')
			self.buildDict()
		self.dictSize = len(self.dictionary.keys())
			
		# build training dataset using dictionary
		if os.path.isfile(self.trainingDataFile) and os.path.isfile(self.trainingLabelsFile) and os.path.isfile(self.countFile) and os.path.isfile(self.categoryMap):
			logging.info('Loading dataset...')
			self.trainingData = np.load(self.trainingDataFile)
			self.trainingLabels = np.load(self.trainingLabelsFile)
			self.categoryCounts = cPickle.load(open(self.countFile))
			self.categories = cPickle.load(open(self.categoryMap))
		else:
			logging.info('Building training dataset...')
			self.buildTrainingSet()

		# reverse category:id mapping
		self.categories = {v:k for k,v in self.categories.iteritems()}

		# priors
		logging.info('Calculating priors...')
		numDocs = sum(self.categoryCounts.values())
		self.categoryCounts = {cat:val/(numDocs*1.0) for cat,val in self.categoryCounts.iteritems()}
		priors = [self.categoryCounts[self.categories[i]] for i in sorted(self.categories.keys())]

		# train classifier
		logging.info('Training classifier...')

		self.classifier = MultinomialNB(class_prior=priors)
		self.classifier.fit(self.trainingData, self.trainingLabels)
		
		logging.info('Classifier trained')

	def buildTrainingSet(self):
		self.categoryCounts = Counter(); self.categoryFeatures = {};

		f = open(self.docsFile)
		count = 0
		for line in f:
			split = line.split('\t')
			categories = split[0].split('***')
			doc = split[1]
			
			for cat in categories:
				# keep track of category counts
				self.categoryCounts[cat] += 1
	
				# track category feature/word counts
				if cat not in self.categoryFeatures:
					self.categoryFeatures[cat] = np.zeros(self.dictSize)
				
				docSparse = self.process(doc.decode('utf8'))
				for k,v in docSparse:
					self.categoryFeatures[cat][k] += v
				
			count += 1
			if count%self.chunkSize == 0: logging.info('(TRAINING) ' + str(count) + ' lines processed.')
		
		# id's for categories
		self.categories = {}
		for key in self.categoryFeatures.keys():
			if key not in self.categories:
				self.categories[key] = len(self.categories)
		
		self.trainingData = np.zeros((len(self.categories.keys()), self.dictSize))
		for cat in self.categories.keys():
			self.trainingData[self.categories[cat]] = self.categoryFeatures[cat]
		self.trainingLabels = np.array(range(0, len(self.trainingData)))  
		
		self.trainingData.dump(self.trainingDataFile)
		self.trainingLabels.dump(self.trainingLabelsFile)
		cPickle.dump(self.categoryCounts, open(self.countFile, 'w+'))
		cPickle.dump(self.categories, open(self.categoryMap, 'w+'))
		
	def buildDict(self):
		f = open(self.docsFile)
		self.dictionary	= gensim.corpora.dictionary.Dictionary()
		count = 0; data = []
		for line in f:
			split = line.split('\t')
			doc = self.processDoc(split[1].decode('utf-8'))
			data.append(doc)
			count += 1
			if count % self.chunkSize == 0: 
				logging.info('(DICTIONARY) ' + str(count) + ' documents processed')
				self.dictionary.add_documents(data)
				del data[:]

		# filter dict extremes
		self.dictionary.filter_extremes(no_below=5, no_above=.3, keep_n=None)
		cPickle.dump(self.dictionary, open(self.dictFile, 'w+'))	

	def classify(self, s):
		return self.categories[self.classifier.predict(self.vectorize(s))[0]]

	def vectorize(self, s):
		x = np.zeros(self.dictSize)
		for i,j in self.process(s):
			x[i] = j
		return x

	def process(self, s):
		return self.dictionary.doc2bow(self.processDoc(s))
		
	def hasNumbers(self, inputString):
        	return any(char.isdigit() for char in inputString)

	def processDoc(self, s):
		#doc = unicodedata.normalize('NFKD', s).encode('ascii','ignore').lower().translate(None, string.punctuation)
		doc = s.lower().translate(None, string.punctuation)	# non-unicode strings
		return [self.lmtzr.lemmatize(e) for e in doc.split() if len(e) > 1 and not self.hasNumbers(e)]

if __name__=="__main__":
	c = TextClassifier(docsFile='records.txt', logFile='classifier.log')
	c.train()
	print c.classify('calculus topology')
