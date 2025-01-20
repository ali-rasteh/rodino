# RoDINO

## Overview
<p align="justify"> Recently, learning task-agnostic representations using foundation models has gotten
great attention in computer vision. Rich representations learned using foundation
models are showing impressive performance on several downstream tasks including
classification, segmentation, Image retrieval, etc. On the other hand, as it’s well
known, Neural Networks are vulnerable to adversarial attacks. The vulnerability of
foundation models to adversarial attacks harms the performance of the model to all
of the downstream tasks, and therefore having a robust representation can have a
huge impact on the robustness of all tasks. Considering this fundamental impact, in
this project, we propose RoDINO (Robust DINO) which is a method to boost the
empirical robustness of downstream tasks by leveraging PGD attack to generate
adversary images and adversarially train DINO which is a self-supervised
representation learning model with Vision Transformers backbone. </p>

## License
This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Commands
### Training RoDINO

Run the following command to perform the adversarial training on the dino and achieve RoDINO:

```
python run_with_submitit.py --nodes 1 --ngpus 4 --arch vit_small --data_path path/to/data --output_dir ./save/ --arch vit_small --patch_size 16 --epochs 100 --batch_size_per_gpu 32  --saveckp_freq 1 
```

To do the fine-tuning, we provided the pre-trained version of dino and performed the adversarial training.

### Training Classifier

```
python -m dino.eval_linear --arch vit_small --patch_size 16 --epochs 100 --num_labels 100 --data_path /data/sara/imagenet-100/ --output_dir ./save/ --pretrained_weights path/to/rodino
```

### Evaluating the Classifier

* Evaluating natural accuracy on RoDINO backbone:
```
python -m dino.eval_linear --arch vit_small --patch_size 16 --epochs 100 --num_labels 100 --data_path /data/sara/imagenet-100/ --output_dir ./save/ --pretrained_weights path/to/rodino --linear_pretrained_weights path/to/rodino_classifier --evaluate
```
* Evaluating adversary accuracy on RoDINO:
```
python ./dino/eval_linear.py --arch vit_small --patch_size 16 --epochs 100 --num_labels 100 --data_path /data/sara/imagenet-100/ --output_dir ./save/ --pretrained_weights path/to/rodino --linear_pretrained_weights path/to/rodino_classifier --evaluate --pgd_attack linf  --pgd_size 0.01
```

### KNN Evaluation

* Evaluating natural accuracy on RoDINO backbone:
```
python eval_knn.py --pretrained_weights path/to/rodino --checkpoint_key teacher --data_path path/to/data
```

* Evaluating adversary accuracy on RoDINO:
```
python eval_knn.py --pretrained_weights path/to/rodino --checkpoint_key teacher --data_path path/to/data --attack linf
```
