import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence

import abc

from PIL import Image

import numpy as np
import os
import json
import datetime
import copy
import logging

from preprocessing.vocabulary import Vocabulary

from config import *

from preprocessing.preprocess_text import preprocess_question_sentence

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S"
        )
logger = logging.getLogger(__name__)

class VQADataset(Dataset, abc.ABC):

	# answer vocabulary
	answers_vocabulary = Vocabulary(a_vocab_path)

	# question vocabulary
	questions_vocabulary = Vocabulary(q_vocab_path)

	# get max question length
	with open(questions_stats_path, "r") as f:
		max_question_length = json.load(f)["max_question_length"]

	@classmethod
	def vqa_collate(cls, batch):
		""" Custom collate function for dataloader """
		mini_batch = dict()
		mini_batch["image"] = torch.stack([item["image"] for item in batch]).float()
		mini_batch["question"] = torch.stack([item["question"] for item in batch]) # indices need to be int
		mini_batch["answer"] = torch.stack([item["answer"] for item in batch]).float()
		mini_batch["question_id"] = torch.tensor([item["question_id"] for item in batch], dtype=int).reshape(-1,1)
		mini_batch["answers"] = [item["answers"] for item in batch]
		return mini_batch

	def __init__(self, question_file):
		# load dataset
		self.questions = json.load(open(question_file, 'r'))

	@abc.abstractmethod
	def __len__(self):
		pass

	@abc.abstractmethod
	def __getitem__(self, index):
		pass

	def preprocess_image(self, img_path):
		""" Helper method to preprocess an image """	
		# always opening images in rgb - so that greyscale images are copy through all 3 channels
		image = Image.open(img_path).convert("RGB")

		# apply transformation on the image
		if self.transform:
			image = self.transform(image)
		
		return image

	def preprocess_question(self, question):
		""" 
		param: question (String): question string
		return: question in bag of words vector 
		"""
		indices = torch.empty(self.max_question_length, dtype=int).fill_(self.questions_vocabulary.word2idx(Vocabulary.PAD_TOKEN))
		words = preprocess_question_sentence(question)
		for i, word in enumerate(words):
			indices[i] = self.questions_vocabulary.word2idx(word)
		return indices


	def info(self):
		"""
		Print information about the VQA annotation file.
		:return:
		"""
		for key, value in self.questions['info'].items():
			logger.info('%s: %s'%(key, value))

class VQA(VQADataset):

	def __init__(self,
				annotation_file: str,
				question_file: str,
				img_dir: str,
				transform=None):
		"""
		Create the VQA Dataset

		:param annotation_file: location of VQA annotation file
		:param question_file: location of VQA question file
		:param img_dir: directory containing all images
		:param transform: optional transform to be applied on an image sample.
		"""
		logger.info('Loading VQA annotations and questions into memory...')
		time_t = datetime.datetime.utcnow()
		super(VQA, self).__init__(question_file)
		self.annotations = json.load(open(annotation_file, 'r'))
		logger.info("Done in {}".format(datetime.datetime.utcnow() - time_t))

		# dictionary mapping question id to the annotation
		self.question_annotation = {annotation['question_id']: [] for annotation in self.annotations['annotations']}

		# dictionary mapping question id to question
		self.questions_id = {annotation['question_id']: [] for annotation in self.annotations['annotations']}

		# dictionary mapping image id to its annotations
		self.img2QA = {annotation['image_id']: [] for annotation in self.annotations['annotations']}

		# create index
		self.create_index()

		# store an array of question ids for indexing
		self.q_ids = list(self.questions_id.keys())

		self.data_subtype = self.questions["data_subtype"]
		self.task_type = self.questions["task_type"]
		self.data_type = self.questions["data_type"]
		self.transform = transform
		self.img_dir = img_dir

		logger.info("Annotation file: %s", annotation_file)
		logger.info("Question file: %s", question_file)
		logger.info("Data type: %s", self.data_type)
		logger.info("Data subtype: %s", self.data_subtype)
		logger.info("Image directory: %s", img_dir)
		logger.info("Task type: %s", self.task_type)
		if transform:
			logger.info("Transform: %s", transform)

	def create_index(self):
		# create index
		logger.info('Creating index...')
		for annotation in self.annotations['annotations']:
			self.img2QA[annotation['image_id']] += [annotation]
			self.question_annotation[annotation['question_id']] = annotation
		for ques in self.questions['questions']:
			self.questions_id[ques['question_id']] = ques
		logger.info('Index created!')

	def __len__(self):
		""" Return length of the dataset based on the number of questions """
		return len(self.questions_id)

	def __getitem__(self, index):
		""" 
		Each sample consist of (question, image, answer).
		"""
		q_id = self.q_ids[index]
		annotation = self.question_annotation[q_id]
		img_id = annotation["image_id"]

		# get the image from disk
		img_name = 'COCO_' + self.data_subtype + '_'+ str(img_id).zfill(12) + '.jpg'
		img_path = os.path.join(self.img_dir,img_name)

		# read image from disk
		if os.path.isfile(img_path):
			image = self.preprocess_image(img_path)
		else:
			logger.error(f"{img_path} is not a valid file.")
			exit(1)

		# get the question
		question = self.questions_id[q_id]["question"]

		# get the question's answer
		multiple_choice_answer = annotation["multiple_choice_answer"]

		question = self.preprocess_question(question)
		answer = self.preprocess_answer(multiple_choice_answer)

		# TODO process answer depending on answer type? # if annotation["answer_type"] == "number": # elif annotation["answer_type"] == "yes/no": # elif annotation["answer_type"] == "other":

		return {"image":image, 
				"question": question, 
				"answer": answer, 
				"question_id": q_id, 
				"answers": [answer["answer"] for answer in annotation["answers"]], 
				"answer_type": annotation["answer_type"], 
				"question_type":annotation["question_type"]}
	
	def preprocess_answer(self, answer):
		""" 
		param: answer (String): answer string
		vocab (dict): vocabulary
		return: answer in one-hot vector 
		"""
		one_hot = torch.zeros(self.answers_vocabulary.size)
		one_hot[self.answers_vocabulary.word2idx(answer)] = 1
		return one_hot

	def get_question_ids(self, imgIds = [], quesTypes = [], ansTypes = []):
		"""
		Get question ids that satisfy given filter conditions. default skips that filter
		:param 	imgIds    (int array)   : get question ids for given imgs
				quesTypes (str array)   : get question ids for given question types
				ansTypes  (str array)   : get question ids for given answer types
		:return:    ids   (int array)   : integer array of question ids
		"""
		imgIds 	  = imgIds    if type(imgIds)    == list else [imgIds]
		quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
		ansTypes  = ansTypes  if type(ansTypes)  == list else [ansTypes]

		if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
			anns = self.annotations['annotations']
		else:
			if not len(imgIds) == 0:
				anns = sum([self.img2QA[imgId] for imgId in imgIds if imgId in self.img2QA], [])
			else:
				anns = self.annotations['annotations']
			anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
			anns = anns if len(ansTypes)  == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
		ids = [ann['question_id'] for ann in anns]
		return ids

	def get_img_ids(self, quesIds=[], quesTypes=[], ansTypes=[]):
		"""
		Get image ids that satisfy given filter conditions. default skips that filter
		:param quesIds   (int array)   : get image ids for given question ids
				quesTypes (str array)   : get image ids for given question types
				ansTypes  (str array)   : get image ids for given answer types
		:return: ids     (int array)   : integer array of image ids
		"""
		quesIds   = quesIds   if type(quesIds)   == list else [quesIds]
		quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
		ansTypes  = ansTypes  if type(ansTypes)  == list else [ansTypes]

		if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
			anns = self.annotations['annotations']
		else:
			if not len(quesIds) == 0:
				anns = sum([self.question_annotation[quesId] for quesId in quesIds if quesId in self.question_annotation],[])
			else:
				anns = self.annotations['annotations']
			anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann['question_type'] in quesTypes]
			anns = anns if len(ansTypes)  == 0 else [ann for ann in anns if ann['answer_type'] in ansTypes]
		ids = [ann['image_id'] for ann in anns]
		return ids

	def load_questions_and_answers(self, ids=[]):
		"""
		Load questions and answers with the specified question ids.
		:param ids (int or int array)       : integer ids specifying question ids
		:return: qa (object array)   		: loaded qa objects
		"""
		if type(ids) == list:
			return [self.question_annotation[q_id] for q_id in ids]
		elif type(ids) == int:
			return [self.question_annotation[ids]]

	def show_questions_and_answers(self, anns):
		"""
		Display the specified questions and answers.
		:param anns (array of object): annotations to display
		:return: None
		"""
		if len(anns) == 0:
			return None
		for ann in anns:
			quesId = ann['question_id']
			logger.info("Question: %s" %(self.questions_id[quesId]['question']))
			for ans in ann['answers']:
				logger.info("Answer %d: %s" %(ans['answer_id'], ans['answer']))

class VQATest(VQADataset):

	def __init__(self,
				question_file: str,
				img_dir: str,
				transform=None):
		"""
		Create the VQA Test Dataset

		:param question_file: location of VQA question file
		:param img_dir: directory containing all images
		:param transform: optional transform to be applied on an image sample.
		"""

		# load dataset
		logger.info('Loading VQA test questions into memory...')
		time_t = datetime.datetime.utcnow()
		super(VQATest, self).__init__(question_file)
		logger.info("Done in {}".format(datetime.datetime.utcnow() - time_t))

		# store an array of question ids for indexing
		self.q_ids = [question["question_id"] for question in self.questions["questions"]]

		# dictionary mapping question id to question
		self.questions_id = {}
		for ques in self.questions['questions']:
			self.questions_id[ques['question_id']] = ques

		self.data_subtype = self.questions["data_subtype"].replace("-dev", "")
		self.task_type = self.questions["task_type"]
		self.data_type = self.questions["data_type"]
		self.transform = transform
		self.img_dir = img_dir

		logger.info("Question file: %s", question_file)
		logger.info("Data type: %s", self.data_type)
		logger.info("Data subtype: %s", self.data_subtype)
		logger.info("Image directory: %s", img_dir)
		logger.info("Task type: %s", self.task_type)
		if transform:
			logger.info("Transform: %s", transform)

	def __len__(self):
		""" Return length of the dataset based on the number of questions """
		return len(self.questions_id)

	def __getitem__(self, index):
		""" 
		Each sample consist of (question, image, answer).
		"""
		q_id = self.q_ids[index]
		question_dict = self.questions_id[q_id]
		question = question_dict["question"]
		img_id = question_dict["image_id"]

		# get the image from disk
		img_name = 'COCO_' + self.data_subtype + '_'+ str(img_id).zfill(12) + '.jpg'
		img_path = os.path.join(self.img_dir,img_name)

		# read image from disk
		if os.path.isfile(img_path):
			image = self.preprocess_image(img_path)
		else:
			logger.error(f"{img_path} is not a valid file.")
			exit(1)

		question = self.preprocess_question(question)

		return {"image":image, "question": question, "question_id": q_id}