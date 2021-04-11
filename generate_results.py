import json

from config import *
from simple_vqa_baseline import SimpleBaselineVQA
from simple_vqa_baseline import preprocess as simple_preprocess
from vqa import VQA

taskType = 'OpenEnded'  # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
dataType = 'mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
dataSubType = 'test2015'
resultType = 'real'

resultFile = f'Results/{taskType}_{dataType}_{dataSubType}_{resultType}_results.json'
results = list()

if __name__ == '__main__':

    test_dataset = VQA(
        test_annFile,
        test_quesFile,
        test_imgDir,
        transform=simple_preprocess
    )

    simplebaseline = SimpleBaselineVQA.load_from_checkpoint('models/SimpleBaselineVQA/80709edb7f9f4b6c90a7905e780becb8/checkpoints/epoch=2-step=41603.ckpt')

    simplebaseline.eval()
    simplebaseline.freeze()

    total_test_points = len(test_dataset)
    for i, (image, question, answer) in test_dataset:
        if i % 1000 == 0:
            print(f"Evaluated {i} out of {total_test_points}")

        results.append({f"{question}": simplebaseline(image, question)})

    with open(resultFile, 'w') as outfile:
        json.dump(results, outfile)
