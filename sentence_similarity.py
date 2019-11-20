import models
import torch
from numpy import dot
from numpy.linalg import norm
from models import InferSent

def cosine_similarity(vec1,vec2):
	assert vec1.shape==vec2.shape

	return dot(vec1, vec2)/(norm(vec1)*norm(vec2))


class SentenceSimilarityScore:
	def __init__(self,threshold=1.0,V=1,K=10000):
		self.threshold = threshold
		MODEL_PATH = 'encoder/infersent%s.pkl' % V
		params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
		                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
		self.model = InferSent(params_model)
		self.model.load_state_dict(torch.load(MODEL_PATH))
		use_cuda = False
		self.model = self.model.cuda() if use_cuda else self.model
		W2V_PATH = 'GloVe/glove.840B.300d.txt' 
		self.model.set_w2v_path(W2V_PATH)
		self.model.build_vocab_k_words(K=K)

	def score(self,sentence_1,sentence_2):
		input_sentence = [sentence_1.strip(),sentence_2.strip()]
		embeddings = model.encode(sentences, bsize=128, tokenize=False, verbose=True)
		return cosine_similarity(embeddings[0],embeddings[-1])





