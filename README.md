This repo shows the source code of EMNLP 2024 paper: [Context-Aware Adapter Tuning for Few-Shot Relation Learning in Knowledge Graphs (RelAdapter)](https://arxiv.org/pdf/2410.09123) framework for few-shot relation learning (FSRL).

![Description of Image](images/framework.jpg)


# Pytorch RelAdapter (Few-shot Link Prediction)

## Evironment Setting
This code is lastly tested with:
* Python 3.6.7
* PyTorch 1.0.1
* tensorboardX 1.8

You can also install dependencies by
```bash
pip install -r requirements.txt
```

## Dataset
We provide three datasets: Wiki[^1], FB15K-237[^1] and UMLS.


## Training (UMLS)
```bash
python main.py  --seed 1  --dataset umls-One  --data_path ./umls    --few 3  --step train  --mu 0.3  --alpha 0.1 --neuron 50    --eval_by_rel False   --prefix umlsone_3shot_pretrain  --device 0
```

## Test (UMLS)
```bash
python main.py  --seed 1  --dataset umls-One  --data_path ./umls    --few 3  --step test  --mu 0.3  --alpha 0.1 --neuron 50    --eval_by_rel True --prefix umlsone_3shot_pretrain  --device 0
```
**Wiki and FB15K-237 follow the same code format for training and test**

Here are explanations of some important args,

```bash
--dataset:   "the name of dataset, Wiki, FB15K-237, UMLS"
--data_path: "directory of dataset"
--few:       "the number of few in {few}-shot, as well as instance number in support set"
--data_form: "dataset setting, Pre-Train"
--alpha : "the adapter ratio"
--mu : "the context ratio"
--neurons : "the adapter neurons"
--prefix:    "given name of current experiment"
--device:    "the GPU number"
```
[^1]: Due to size constraint, Wiki and FB15K-237 have been excluded.



