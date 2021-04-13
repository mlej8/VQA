import json

from config import *
from simple_vqa_baseline import SimpleBaselineVQA
from simple_vqa_baseline import preprocess as simple_preprocess
from vqa_cnn_lstm import OriginalVQA
from vqa_cnn_lstm import preprocess as original_preprocess
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
        transform=original_preprocess
    )

    original_model = OriginalVQA.load_from_checkpoint('models/OriginalVQA/4a618c13c47a4fd3955355bfd9546c41/checkpoints/epoch=2-step=41603.ckpt')

    original_model.eval()
    original_model.freeze()

    total_test_points = len(test_dataset)
    for i, data_dict in test_dataset:
        if i % 1000 == 0:
            print(f"Evaluated {i} out of {total_test_points}")

        results.append({data_dict["question_id"]: original_model(data_dict["image"], data_dict["question"])})

    with open(resultFile, 'w') as outfile:
        json.dump(results, outfile)
