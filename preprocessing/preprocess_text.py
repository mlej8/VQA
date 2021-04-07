  
import os
import argparse
import numpy as np
import json
import re
from collections import defaultdict
import logging

import nltk
from nltk.tokenize import word_tokenize

from config import * 

import datetime

nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_questions_vocabulary(questions_dir):
    """
    Make dictionary for all questions and save them into text file.
    
    :param questions_dir: directory to all question datasets.
    """
    logger.info('Making vocabulary for all questions.')

    # tracking time
    start_time = datetime.datetime.utcnow()

    # vocabulary
    vocabulary = set()

    # storing all questions' length
    q_len = []

    # get all questions datasets
    datasets = os.listdir(questions_dir)
    
    logger.info('Using the following questions datasets:')
    for dataset in datasets:
        logger.info(dataset)

    for dataset in datasets:    
        with open(os.path.join(questions_dir, dataset)) as f:
            # get all the questions
            questions = json.load(f)['questions']
        
        for question_obj in questions:
            # get a question without punctuation
            question = question_obj["question"].lower()[:-1]

            # tokenize sentence 
            words = word_tokenize(question)
            
            # update the words
            vocabulary.update(words)

            # track questions length
            q_len.append(len(words))

        logger.info('Done: %s', dataset)

    # add start and end token
    vocabulary = list(vocabulary)
    
    # max question length
    max_len_q = max(q_len)

    # make sure that vocabulary dir exists
    vocabulary_dir = os.path.join(dataDir, "Vocabulary")
    if not os.path.exists(vocabulary_dir):
        os.makedirs(vocabulary_dir)

    # create questions vocabulary
    questions_vocabulary_path = os.path.join(vocabulary_dir, "questions_vocabulary.txt")
    with open(questions_vocabulary_path, 'w') as f:
        f.write("\n".join(vocabulary))
    logger.info('Vocabulary file for questions written at: %s', questions_vocabulary_path)
    logger.info('The total number of words for the questions vocabulary: %d' % len(vocabulary))
    logger.info('Maximum length of a question: %d' % max_len_q)
    
    max_length_path = os.path.join(vocabulary_dir, "max_question_length.txt")
    with open(max_length_path, 'w') as f:
        f.write(str(max_len_q))

    logger.info('Done building questions vocabulary in %0.2fs.' % ((datetime.datetime.utcnow() - start_time).total_seconds()))

# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))

# TODO understand how process_punctuation
# def process_punctuation(s):
#     # The only normalization that is applied to both machine generated answers as well as
#     # ground truth answers is replacing most punctuation with space (see [0] and [1]).
#     # Since potential machine generated answers are just taken from most common answers, applying the other
#     # normalizations is not needed, assuming that the human answers are already normalized.
#     # [0]: http://visualqa.org/evaluation.html
#     # [1]: https://github.com/VT-vision-lab/VQA/blob/3849b1eae04a0ffd83f56ad6f70ebd0767e09e0f/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L96
    
#     if _punctuation.search(s) is None:
#         return s
#     s = _punctuation_with_a_space.sub('', s)
#     if re.search(_comma_strip, s) is not None:
#         s = s.replace(',', '')
#     s = _punctuation.sub(' ', s)
#     s = _period_strip.sub('', s)
#     return s.strip()

def make_answers_vocabulary(annotations_dir, n_answers=1000):
    """ 
    Make vocabulary for top `n_answers` most frequent answers and save them into a text file.
    
    :param questions_dir str: directory to all annotation datasets.
    :param n_answers int: include n_answers most frequent answers.
    """
    logger.info('Making vocabulary for all answers.')

    # tracking time
    start_time = datetime.datetime.utcnow()

    # default dict returns 0 as default value
    answers = defaultdict(lambda: 0)
    
    # list all the annotation files
    datasets = os.listdir(annotations_dir)
    logger.info('Using the following annotation datasets:')
    for dataset in datasets:
        logger.info(dataset)

    # for each dataset, extract the possible answers
    for dataset in datasets:
        with open(os.path.join(annotations_dir, dataset)) as f:
            annotations = json.load(f)['annotations']
        
        # extract answers for all annotations
        for annotation in annotations:
            for answer in annotation['answers']:
                answers[answer['answer'].lower()] += 1
    
        logger.info('Done: %s', dataset)
    
    # sort by descending in count first, then use lexicographical order
    answers = sorted(answers, key=lambda x: (answers[x], x), reverse=True)
    
    # make sure the end token is not in answers
    if n_answers > len(answers):
        top_answers = answers
    else: 
        top_answers = answers[:n_answers]

    # make sure that vocabulary dir exists
    vocabulary_dir = os.path.join(dataDir, "Vocabulary")
    if not os.path.exists(vocabulary_dir):
        os.makedirs(vocabulary_dir)

    # create answers vocabulary
    answers_vocabulary_path = os.path.join(vocabulary_dir, "answers_vocabulary.txt")
    with open(answers_vocabulary_path, 'w') as f:
        f.write('\n'.join(top_answers))

    logger.info('Total number of possible answers: %d' % len(answers))
    logger.info(f'Preserved top {n_answers} answers.')
    logger.info('Done building answers vocabulary in %0.2fs.' % ((datetime.datetime.utcnow() - start_time).total_seconds()))


def main(args):
    make_questions_vocabulary(os.path.join(args.dataset_dir, "Questions"))
    make_answers_vocabulary(os.path.join(args.dataset_dir, "Annotations"), args.num_answers)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default="datasets",
                        help='Parent directory to directories containing input questions and annotations')
    parser.add_argument('--num_answers', type=int, default=1000,
                        help='Number of answers to be included in vocabulary. Defaults to 1000.')
    args = parser.parse_args()
    main(args)