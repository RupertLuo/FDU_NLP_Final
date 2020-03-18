import logging
import os
from logging import Logger

import torch
from transformers import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import numpy as np
import json
import pandas as pd
from conceptnet_zc.loading import running
from gensim.models import KeyedVectors
import torch.utils.data as data

logger: Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# 使用FileHandler输出到文件
fh = logging.FileHandler("./logfile/logBERT.log")
fh.setLevel(logging.INFO)
# 使用StreamHandler输出到屏幕
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# 添加两个Handler
logger.addHandler(ch)
logger.addHandler(fh)


def record_wrong(wrong_samples, args):
	test_file = os.path.join(args.data_dir, "test" + args.data_name + ".tsv")
	dataset = pd.read_csv(test_file, sep='\t', header=None)

	records = pd.DataFrame()
	for index in wrong_samples:
		line = dataset.loc[dataset[0] == index]
		records = records.append(line)

	save_path = args.wrong_file
	records.to_csv(save_path, sep='\t', index=False, header=False)


def simple_accuracy(preds, labels):
	# print("preds", len(preds))
	# print("labels", len(labels))
	assert len(preds) == len(labels)
	# labels = labels.detach().cpu().numpy()
	# preds = preds.int().numpy()
	return (preds == labels).mean()


class InputExample(object):
	"""
	A single training/test example for simple sequence classification.
	"""

	def __init__(self, guid, sent0_a, sent1_a, sent0_b=None, sent1_b=None, sent0_c=None, sent1_c=None, label=None):
		self.guid = guid
		self.sent0_a = sent0_a
		self.sent1_a = sent1_a
		self.sent0_b = sent0_b
		self.sent1_b = sent1_b
		self.sent0_c = sent0_c
		self.sent1_c = sent1_c
		self.label = label


class Task1Processor(DataProcessor):
	"""
	Processor for data converters for sequence classificaiton datasets in Task1.
	Datafile -> Examples
	"""

	def get_train_examples(self, data_dir, data_name):
		data_name = "train" + data_name + ".tsv"
		logger.info("Looking at {}".format(os.path.join(data_dir, data_name)))
		return self._create_examples(lines=self._read_tsv(os.path.join(data_dir, data_name)))

	def get_dev_examples(self, data_dir, data_name):
		data_name = "dev" + data_name + ".tsv"
		logger.info("Looking at {}".format(os.path.join(data_dir, data_name)))
		return self._create_examples(lines=self._read_tsv(os.path.join(data_dir, data_name)))

	def get_test_examples(self, data_dir, data_name):
		data_name = "test" + data_name + ".tsv"
		print('Im printing "Looking at {}".format(os.path.join(data_dir, data_name)')
		logger.info("Looking at {}".format(os.path.join(data_dir, data_name)))
		return self._create_examples(lines=self._read_tsv(os.path.join(data_dir, data_name)))

	def get_labels(self):
		return ["0", "1"]

	def _create_examples(self, lines):
		# Datafile -> Examples
		examples = []
		for (i, line) in enumerate(lines):
			# guid = "%s-%s" % (set_type, i)
			guid = int(line[0])

			# # s0 [SEP] w0   VS   s1 [SEP] w1
			# sent0_a = line[3]
			# sent1_a = line[4]
			# sent0_b = None if len(line) == 5 else line[5]
			# sent1_b = None if len(line) == 5 else line[6]

			# # s0 [SEP] e   VS   s1 [SEP] e
			# sent0_a = line[3]
			# sent1_a = line[4]
			# sent0_b = line[7]
			# sent1_b = line[7]

			# exp [SEP] s0 VS exp [SEP] s1
			# sent0_a = line[7]
			# sent1_a = line[7]
			# sent0_b = line[3]
			# sent1_b = line[4]

			# # s0 [SEP] w0 [SEP] exp VS s1 [SEP] w1 [SEP] exp
			# sent0_a = line[3]
			# sent1_a = line[4]
			# sent0_b = line[5]
			# sent1_b = line[6]
			# sent0_c = line[7]
			# sent1_c = line[7]

			# s0 [SEP] exp [SEP] w0 VS s1 [SEP] exp [SEP] w1
			sent0_a = line[3]
			sent1_a = line[4]
			sent0_b = line[7]
			sent1_b = line[7]
			# sent0_b = ''
			# sent1_b = ''
			sent0_c = line[5]
			sent1_c = line[6]
			# sent0_c = ''
			# sent1_c = ''

			label = line[1]
			examples.append(InputExample(guid=guid,
			                             sent0_a=sent0_a, sent0_b=sent0_b, sent0_c=sent0_c,
			                             sent1_a=sent1_a, sent1_b=sent1_b, sent1_c=sent1_c,
			                             label=label))
		return examples


class InputFeatures(object):
	def __init__(self, guid, sents_features, label):
		self.guid = guid
		self.sents_features = [
			{
				"input_ids": input_ids,
				"attention_mask": attention_mask,
				"token_type_ids": token_type_ids
			}
			for input_ids, attention_mask, token_type_ids in sents_features
		]
		self.label = label


def convert_examples_to_features(examples, tokenizer,
                                 max_length=218,
                                 label_list=None,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 mask_padding=0):
	"""
	Loads a data file into a list of 'InputFeatures'.
	return:
		a list of task-specific ``InputFeatures`` which can be fed to the model.
	Examples -> Features example属性中的sent0/sent1做了encoding
	"""
	logger.info("Using label list %s" % (label_list))
	label_map = {label: i for i, label in enumerate(label_list)}
	features = []

	for (ex_index, example) in enumerate(examples):
		if ex_index % 1000 == 0:
			logger.info("Writing example %d" % ex_index)
		sents_features = []
		sent0_a, sent1_a = example.sent0_a, example.sent1_a
		sent0_b, sent1_b = example.sent0_b, example.sent1_b
		sent0_c, sent1_c = example.sent0_c, example.sent1_c

		for (text_a, text_b, text_c) in ((sent0_a, sent0_b, sent0_c), (sent1_a, sent1_b, sent1_c)):
			if sent0_c is None:
				tokens_a = tokenizer.tokenize(text_a)
				tokens_b = None
				if text_b:
					tokens_b = tokenizer.tokenize(text_b)
					_truncate_seq_pair(tokens_a, tokens_b, max_length - 3)
				else:
					# Account for [CLS] and [SEP] with "- 2"
					if len(tokens_a) > max_length - 2:
						tokens_a = tokens_a[:(max_length - 2)]

				# The convention in BERT is:
				# (a) For sequence pairs:
				#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
				#  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
				# (b) For single sequences:
				#  tokens:   [CLS] the dog is hairy . [SEP]
				#  type_ids:   0   0   0   0  0     0   0
				#
				# Where "type_ids" are used to indicate whether this is the first
				# sequence or the second sequence. The embedding vectors for `type=0` and
				# `type=1` were learned during pre-training and are added to the wordpiece
				# embedding vector (and position vector). This is not *strictly* necessary
				# since the [SEP] token unambiguously separates the sequences, but it makes
				# it easier for the model to learn the concept of sequences.
				#
				# For classification tasks, the first vector (corresponding to [CLS]) is
				# used as as the "sentence vector". Note that this only makes sense because
				# the entire model is fine-tuned.
				tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
				token_type_ids = [0] * len(tokens)

				if tokens_b:
					tokens += tokens_b + ['[SEP]']
					token_type_ids += [1] * (len(tokens_b) + 1)
			else:
				# Save the truncate step
				tokens_a = tokenizer.tokenize(text_a)
				tokens_b = tokenizer.tokenize(text_b)
				tokens_c = tokenizer.tokenize(text_c)
				tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
				token_type_ids = [0] * len(tokens)
				tokens += tokens_b + ["[SEP]"]
				token_type_ids += [1] * (len(tokens_b) + 1)
				tokens += tokens_c + ["[SEP]"]
				token_type_ids += [0] * (len(tokens_c) + 1)

			input_ids = tokenizer.convert_tokens_to_ids(tokens)

			# The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
			attention_mask = [1] * len(input_ids)
			# Zero-pad up to the sequence length.
			padding_length = max_length - len(input_ids)
			input_ids = input_ids + ([pad_token] * padding_length)
			attention_mask = attention_mask + ([mask_padding] * padding_length)
			token_type_ids = token_type_ids + ([
				                                   pad_token_segment_id] * padding_length)  # Segment token indices to indicate first and second portions of the inputs.

			assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
			assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
			                                                                                    max_length)
			assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
			                                                                                    max_length)
			# 两句话的features各自储存为tuple
			sents_features.append((input_ids, attention_mask, token_type_ids))

			# tokens = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=max_length)
			#
			# input_ids, token_type_ids = tokens["input_ids"], tokens["token_type_ids"]
			# # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
			# attention_mask = [1] * len(input_ids)
			# # Zero-pad up to the sequence length.
			# padding_length = max_length - len(input_ids)
			# input_ids = input_ids + ([pad_token] * padding_length)
			# attention_mask = attention_mask + ([mask_padding] * padding_length)
			# token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)  # Segment token indices to indicate first and second portions of the inputs.
			#
			# assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
			# assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
			# assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)
			# # 两句话的features各自储存为tuple
			# sents_features.append((input_ids, attention_mask, token_type_ids))
		label = label_map[example.label]

		if ex_index < 3:
			logger.info("*** Example ***")
			logger.info("tokens: %s" % (tokens))
			logger.info("guid: %s" % (example.guid))
			logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))

			logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
			logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
			logger.info("label: %s (id = %d)" % (example.label, label))

		features.append(
			InputFeatures(
				guid=example.guid,
				sents_features=sents_features,
				label=label
			)
		)
	return features


class Task1DataLoader(data.Dataset):
	def __init__(self, sent_trp_jsonl, batch_size, n_devices, tokenizer, data_name,
	             shuffle=True, start=0, end=None, cut_off=3,
	             is_test=False, use_cache=True, max_seq_length=95, max_trp_length=30, dev=False, test=False):
		super(Task1DataLoader, self).__init__()
		self.batch_size = batch_size
		self.n_devices = n_devices
		self.is_test = is_test

		self.sent_input_ids, self.sent_attention_mask, self.sent_token_type_ids, \
		self.correct_labels, self.Totaladjmatrix, self.word_embedding, self.Referlist = \
			self._load_data(sent_trp_jsonl, tokenizer, max_seq_length, max_trp_length, data_name, dev, test)

		for name in ['sent_input_ids', 'sent_attention_mask', 'sent_token_type_ids',
		             'correct_labels', 'Totaladjmatrix', 'word_embedding', 'Referlist']:
			obj = getattr(self, name)
			setattr(self, name, obj[start:end])

		assert len(self.sent_input_ids) == len(self.sent_attention_mask) == len(self.sent_token_type_ids) \
		       == len(self.correct_labels) == len(self.Totaladjmatrix) == len(self.word_embedding) == len(
			self.Referlist)
		self.n_samples = len(self.correct_labels)

		if shuffle:
			self.permutation = torch.randperm(self.n_samples)
		else:
			self.permutation = torch.arange(self.n_samples)

	def _load_data(self, data_dir, tokenizer, max_seq_length, max_trp_length, data_name, dev=False, test=False):
		# transfer string to bert input format
		# only q[SEP]a , a_trp
		processor = Task1Processor()
		label_list = processor.get_labels()

		if dev:
			cached_mode = "dev"
		elif test:
			cached_mode = "test"
		else:
			cached_mode = "train"

		# Load data features from cache or dataset file
		cached_features_file = os.path.join(data_dir, "cached_{}_{}".format(cached_mode, str(max_seq_length)))
		# if os.path.exists(cached_features_file):
		#     features = torch.load(cached_features_file)
		# else:
		logger.info("Creating features from dataset file at %s", data_dir)
		# 1) Datafile -> Examples
		if dev:
			examples = processor.get_dev_examples(data_dir=data_dir, data_name=data_name)
		elif test:
			examples = processor.get_test_examples(data_dir=data_dir, data_name=data_name)
		else:
			examples = processor.get_train_examples(data_dir=data_dir, data_name=data_name)

		if dev:
			Totaladjmatrix, Indexlist, Referlist = running('dev')
		elif test:
			Totaladjmatrix, Indexlist, Referlist = running('test')
		else:
			Totaladjmatrix, Indexlist, Referlist = running('train')

		# 2) Examples -> Features
		features = convert_examples_to_features(examples=examples, tokenizer=tokenizer,
		                                        label_list=label_list)  # 传进去所有的data
		torch.save(features, cached_features_file)

		# 3) Features -> Dataset
		# 3.1) Convert to Tensors
		all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)  # [8000, 2, 218]
		all_attention_mask = torch.tensor(select_field(features, "attention_mask"), dtype=torch.long)
		all_token_type_ids = torch.tensor(select_field(features, "token_type_ids"), dtype=torch.long)
		all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
		all_guids = torch.tensor([f.guid for f in features], dtype=torch.long)

		# 把单词转换为向量
		tmp_file = './model/glove/glove.txt'
		model = KeyedVectors.load_word2vec_format(tmp_file)

		word_embedding = []
		for word_list in Indexlist:
			word_embedding_single = []
			for word in word_list:
				if word in model:
					word_embedding_single.append(model[word])
				else:
					word_embedding_single.append(np.zeros(100))
			if len(word_embedding_single) != 0:
				word_embedding_single = torch.tensor(np.array(word_embedding_single))
			word_embedding.append(word_embedding_single)
			for i in range(len(Totaladjmatrix)):
				Totaladjmatrix[i] = torch.tensor(Totaladjmatrix[i])

		for i in range(len(Referlist)):
			for j in range(len(Referlist[i])):
				Referlist[i][j] = torch.tensor(Referlist[i][j])

		return all_input_ids, all_attention_mask, all_token_type_ids, all_labels, Totaladjmatrix, \
		       word_embedding, Referlist

	def __len__(self):
		return self.n_samples

	def __getitem__(self, index):
		return self.Totaladjmatrix[index], self.word_embedding[index], \
		       self.Referlist[index]

	def __iter__(self):
		def to_device(obj, dev_cur):
			if isinstance(obj, (tuple, list)):
				return [to_device(item, dev_cur) for item in obj]
			else:
				return obj.to(dev_cur)

		n_gpu = len(self.n_devices)
		for i in range(0, self.n_samples, self.batch_size):
			j = min(self.n_samples, i + self.batch_size)

			# i: j all_gpu batch number, split into three batch indexes
			indexes = self.permutation[i:j]

			n_batches = []
			split_indexes = [indexes[i: i + int(self.batch_size / n_gpu)] for i in
			                 range(0, len(indexes), int(self.batch_size / n_gpu))]
			for device_cur, index_cur in zip(self.n_devices, split_indexes):
				labels = to_device(self.correct_labels[index_cur], device_cur)
				adjmatrix = to_device([self.Totaladjmatrix[idx] for idx in index_cur], device_cur)
				ReferL = to_device([self.Referlist[idx] for idx in index_cur], device_cur)
				ori_emb = to_device([self.word_embedding[idx] for idx in index_cur], device_cur)

				sent_input_ids = to_device([self.sent_input_ids[idx] for idx in index_cur], device_cur)
				sent_attention_mask = to_device([self.sent_attention_mask[idx] for idx in index_cur], device_cur)
				sent_token_type_ids = to_device([self.sent_token_type_ids[idx] for idx in index_cur], device_cur)

				# trp_input_ids = to_device([self.trp_input_ids[idx] for idx in index_cur], device_cur)
				# trp_attention_mask = to_device([self.trp_attention_mask[idx] for idx in index_cur], device_cur)
				# trp_token_type_ids = to_device([self.trp_token_type_ids[idx] for idx in index_cur], device_cur)

				batch_cur = (labels, adjmatrix, ori_emb, ReferL,
				             sent_input_ids, sent_attention_mask, sent_token_type_ids)

				n_batches.append(batch_cur)

			# yield list(zip(n_batches[0], n_batches[1], n_batches[2], n_batches[3]))

			res = []
			for feat_id in range(len(n_batches[0])):
				tmp = [n_batches[batch_id][feat_id] for batch_id in range(len(n_batches))]
				res.append(tmp)
			yield res

	def reshuffle(self):
		self.permutation = torch.randperm(self.n_samples)


def load_examples_and_cache_features(data_dir, data_name, tokenizer, max_seq_length=128, dev=False, test=False):
	processor = Task1Processor()
	label_list = processor.get_labels()

	if dev:
		cached_mode = "dev"
	elif test:
		cached_mode = "test"
	else:
		cached_mode = "train"

	# Load data features from cache or dataset file
	cached_features_file = os.path.join(data_dir, "cached_{}_{}".format(cached_mode, str(max_seq_length)))
	# if os.path.exists(cached_features_file):
	#     features = torch.load(cached_features_file)
	# else:
	logger.info("Creating features from dataset file at %s", data_dir)
	# 1) Datafile -> Examples
	if dev:
		examples = processor.get_dev_examples(data_dir=data_dir, data_name=data_name)
	elif test:
		examples = processor.get_test_examples(data_dir=data_dir, data_name=data_name)
	else:
		examples = processor.get_train_examples(data_dir=data_dir, data_name=data_name)

	if dev:
		Totaladjmatrix, Indexlist, Referlist = running('dev')
	elif test:
		Totaladjmatrix, Indexlist, Referlist = running('test')
	else:
		Totaladjmatrix, Indexlist, Referlist = running('train')

	# 2) Examples -> Features
	features = convert_examples_to_features(examples=examples, tokenizer=tokenizer, label_list=label_list)  # 传进去所有的data
	torch.save(features, cached_features_file)

	# 3) Features -> Dataset
	# 3.1) Convert to Tensors
	all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)  # [8000, 2, 218]
	all_attention_mask = torch.tensor(select_field(features, "attention_mask"), dtype=torch.long)
	all_token_type_ids = torch.tensor(select_field(features, "token_type_ids"), dtype=torch.long)
	all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
	all_guids = torch.tensor([f.guid for f in features], dtype=torch.long)

	# 3.2) Build dataset
	dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_guids)

	# 把单词转换为向量
	tmp_file = 'glove.txt'
	model = KeyedVectors.load_word2vec_format(tmp_file)

	word_embedding = []
	for word_list in Indexlist:
		word_embedding_single = []
		for word in word_list:
			if word in model:
				word_embedding_single.append(model[word])
			else:
				word_embedding_single.append(np.zeros(100))
		if len(word_embedding_single) != 0:
			word_embedding_single = torch.tensor(np.array(word_embedding_single))
		word_embedding.append(word_embedding_single)

	print('lglglgllg\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
	dataset = (
	all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_guids, Totaladjmatrix, Indexlist, Referlist)
	return dataset


def select_field(features, field):
	return [
		[
			sent[field]
			for sent in feature.sents_features
		]
		for feature in features
	]


def preds_decode(logits):
	"""
	原始labels 0：sent0错误， 1：sent1错误
	preds [[sent0的label=0正确的概率， sent0的label=1错误的概率], [sent1的label=0正确的概率，sent1的label=1错误的概率]]
	"""
	# logits [[batch,2], [batch,2]]
	preds = []
	# loop over batch
	for i in range(logits[0].size(0)):
		sent0 = logits[0][i].squeeze()[1]
		sent1 = logits[1][i].squeeze()[1]
		preds.append(0 if sent0 >= sent1 else 1)
	return torch.Tensor(preds)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.

	# However, since we'd better not to remove tokens of options and questions, you can choose to use a bigger
	# length or only pop from context
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			logger.info('Attention! you are removing from token_b (swag task is ok). '
			            'If you are training ARC and RACE (you are poping question + options), '
			            'you need to try to use a bigger max seq length!')
			tokens_b.pop()
