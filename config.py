import os

# setting the seed for reproducability (it is important to set seed when using DPP mode)
import torch
torch.manual_seed(0)


# configuration variables for VQA task
dataDir		        = 'datasets'
versionType         = 'v2_'          # this should be '' when using VQA v1.0 dataset
taskType            = 'OpenEnded'    # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType            = 'mscoco'       # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v2.0.

# configuration variables for VQA train
train_dataSubType   = 'train2014'
train_annFile       = '%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, train_dataSubType)
train_quesFile      = '%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, train_dataSubType)
train_imgDir 	    = '%s/Images/%s/%s/' %(dataDir, dataType, train_dataSubType)

# configuration variables for VQA validation
val_dataSubType     = 'val2014'
val_annFile         = '%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, val_dataSubType)
val_quesFile        = '%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, val_dataSubType)
val_imgDir 		    = '%s/Images/%s/%s/' %(dataDir, dataType, val_dataSubType)

# configuration variables for VQA test
testdev_dataSubType = 'test-dev2015'
testdev_quesFile    = '%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, testdev_dataSubType)
test_dataSubType    = 'test2015'
test_quesFile       = '%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, test_dataSubType)
test_imgDir 		= '%s/Images/%s/%s/' %(dataDir, dataType, test_dataSubType)

# vocabularies
q_vocab_path = os.path.join(dataDir,"Vocabulary","questions_vocabulary.txt")
a_vocab_path = os.path.join(dataDir, "Vocabulary", "answers_vocabulary.txt")

# stats about questions
questions_stats_path = os.path.join(dataDir,"Vocabulary", "questions_stats.txt")

# preprocessed datasets path
preprocessed_train = os.path.join(dataDir, "preprocessed_train.json")
preprocessed_val = os.path.join(dataDir, "preprocessed_val.json")
preprocessed_test_standard = os.path.join(dataDir, "preprocessed_test_standard.json")
preprocessed_test_dev = os.path.join(dataDir, "preprocessed_test_dev.json")
preprocessed_train_val = os.path.join(dataDir, "preprocessed_train_val.json")
preprocessed_test = os.path.join(dataDir, "preprocessed_test.json")