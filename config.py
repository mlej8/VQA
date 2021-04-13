# setting the seed for reproducability (it is important to set seed when using DPP mode)
from pytorch_lightning.utilities.seed import seed_everything
# setting the seed for reproducability (it is important to set seed when using DPP mode)
seed_everything(7)

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
test_dataSubType    = 'test2015'
test_annFile        = '%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, test_dataSubType)
test_quesFile       = '%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, test_dataSubType)
test_imgDir 		= '%s/Images/%s/%s/' %(dataDir, dataType, test_dataSubType)

# vocabularies
q_vocab_path = "datasets/Vocabulary/questions_vocabulary.txt" 
a_vocab_path = "datasets/Vocabulary/answers_vocabulary.txt"

# stats about questions
questions_stats_path = "datasets/Vocabulary/questions_stats.txt"