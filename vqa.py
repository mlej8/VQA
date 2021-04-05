import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from skimage import io

import os

import json
import datetime
import copy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VQA(Dataset):
	def __init__(self, annotation_file: str, question_file: str, data_subtype: str, img_dir: str, transform=None):
		"""
		Create the VQA Dataset

		:param annotation_file: location of VQA annotation file
		:param question_file: location of VQA question file
		:param data_subtype: can only be 'train2014', 'val2014' or 'test2014'
		:param img_dir: directory containing all images
		:param transform: optional transform to be applied on an image sample.
		"""
		self.data_subtype = data_subtype
		self.transform = transform
		
		# answer vocabulary
		self.ans_vocab = {}

		# question vocabulary
		self.q_vocab = {}

		# load dataset
		logger.info('Loading VQA annotations and questions into memory...')
		time_t = datetime.datetime.utcnow()
		self.annotations = json.load(open(annotation_file, 'r'))
		self.questions = json.load(open(question_file, 'r'))
		logger.info("Done in {}".format(datetime.datetime.utcnow() - time_t))
			
		# dictionary mapping question id to the question annotation
		self.question_annotation = {annotation['question_id']: [] for annotation in self.annotations['annotations']}
		
		# dictionary mapping question id to question
		self.questions_id = {annotation['question_id']: [] for annotation in self.annotations['annotations']}

		# dictionary mapping image id to its annotations
		self.img2QA = {annotation['image_id']: [] for annotation in self.annotations['annotations']}

		# create index
		self.create_index()

		# store an array of image ids for indexing
		self.q_ids = self.questions_id.keys()

		# TODO group questions by answer_type

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
		Eaach sample consist of (question, image, answer).
		"""
		q_id = self.q_ids[index]
		annotation = self.question_annotation[q_id]
		img_id = annotation["image_id"]

		# get the image from disk
		img_name = 'COCO_' + self.data_subtype + '_'+ str(img_id).zfill(12) + '.jpg'
		img_path = os.path.join(self.img_dir,img_name)

		# read image from disk
		if os.path.isfile(img_path):
			image = io.imread(img_path)
		else:
			logger.error(f"{img_path} is not a valid file.")

		# apply transformation on the image
		if self.transform:
			image = self.transform(image)

		# get the question 
		question = annotation[q_id]
		
		# TODO answer depending on answer type
		# if annotation["answer_type"] == "number":
		# elif annotation["answer_type"] == "yes/no":
		# elif annotation["answer_type"] == "other":
		# most frequent ground truth answer
		most_freq_ans = annotation["multiple_choice_answer"]
		answers = annotation["answers"]

		return (question, image, most_freq_ans, answers)

	def info(self):
		"""
		Print information about the VQA annotation file.
		:return:
		"""
		for key, value in self.datset['info'].items():
			logger.info('%s: %s'%(key, value))

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
		
	def load_result(self, resFile, question_file):
		"""
		Load result file and return a result object.
		:param   resFile (str)     : file name of result file
		:return: res (obj)         : result api object
		"""
		res = VQA()
		res.questions = json.load(open(question_file))
		res.annotations['info'] = copy.deepcopy(self.questions['info'])
		res.annotations['task_type'] = copy.deepcopy(self.questions['task_type'])
		res.annotations['data_type'] = copy.deepcopy(self.questions['data_type'])
		res.annotations['data_subtype'] = copy.deepcopy(self.questions['data_subtype'])
		res.annotations['license'] = copy.deepcopy(self.questions['license'])

		logger.info('Loading and preparing results...     ')
		time_t = datetime.datetime.utcnow()
		anns    = json.load(open(resFile))
		assert type(anns) == list, 'results is not an array of objects'
		annsQuesIds = [ann['question_id'] for ann in anns]
		assert set(annsQuesIds) == set(self.get_question_ids()), \
		'Results do not correspond to current VQA set. Either the results do not have predictions for all question ids in annotation file or there is atleast one question id that does not belong to the question ids in the annotation file.'
		for ann in anns:
			quesId 			     = ann['question_id']
			if res.annotations['task_type'] == 'Multiple Choice':
				assert ann['answer'] in self.questions_id[quesId]['multiple_choices'], 'predicted answer is not one of the multiple choices'
			qaAnn                = self.question_annotation[quesId]
			ann['image_id']      = qaAnn['image_id'] 
			ann['question_type'] = qaAnn['question_type']
			ann['answer_type']   = qaAnn['answer_type']
		logger.info('DONE (t=%0.2fs)'%((datetime.datetime.utcnow() - time_t).total_seconds()))

		res.annotations['annotations'] = anns
		res.create_index()
		return res
