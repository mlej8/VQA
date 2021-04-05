# coding: utf-8

from vqa import VQA
import random
import skimage.io as io
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
	dataDir		='datasets'
	versionType ='v2_' # this should be '' when using VQA v2.0 dataset
	taskType    ='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
	dataType    ='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v2.0.
	dataSubType ='train2014'
	annFile     ='%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
	quesFile    ='%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
	imgDir 		= '%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)

	# initialize VQA api for QA annotations
	vqa=VQA(annFile, quesFile, dataSubType, imgDir) 

	# load and display QA annotations for given question types
	"""
	All possible quesTypes for abstract and mscoco has been provided in respective text files in ../QuestionTypes/ folder.
	"""
	annIds = vqa.get_question_ids(quesTypes='how many');   
	anns = vqa.load_questions_and_answers(annIds)
	randomAnn = random.choice(anns)
	vqa.show_questions_and_answers([randomAnn])
	imgId = randomAnn['image_id']
	imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
	if os.path.isfile(imgDir + imgFilename):
		I = io.imread(imgDir + imgFilename)
		plt.imshow(I)
		plt.axis('off')
		plt.show()

	# load and display QA annotations for given answer types
	"""
	ansTypes can be one of the following
	yes/no
	number
	other
	"""
	annIds = vqa.get_question_ids(ansTypes='yes/no');   
	anns = vqa.load_questions_and_answers(annIds)
	randomAnn = random.choice(anns)
	vqa.show_questions_and_answers([randomAnn])
	imgId = randomAnn['image_id']
	imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
	if os.path.isfile(imgDir + imgFilename):
		I = io.imread(imgDir + imgFilename)
		plt.imshow(I)
		plt.axis('off')
		plt.show()

	# load and display QA annotations for given images
	"""
	Usage: vqa.get_img_ids(quesIds=[], quesTypes=[], ansTypes=[])
	Above method can be used to retrieve imageIds for given question Ids or given question types or given answer types.
	"""
	ids = vqa.get_img_ids()
	annIds = vqa.get_question_ids(imgIds=random.sample(ids,5));  
	anns = vqa.load_questions_and_answers(annIds)
	randomAnn = random.choice(anns)
	vqa.show_questions_and_answers([randomAnn])  
	imgId = randomAnn['image_id']
	imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
	if os.path.isfile(imgDir + imgFilename):
		I = io.imread(imgDir + imgFilename)
		plt.imshow(I)
		plt.axis('off')
		plt.show()

