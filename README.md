<h1 align="center">
  FullTextKP
</h1>

<h4 align="center">Keyphrase Generation Beyond the Boundaries of Title and Abstract</h4>

<p align="center">
  <a href="https://2022.emnlp.org/"><img src="https://img.shields.io/badge/Findings%20of%20EMNLP-2022-green"></a>
  <a href="https://arxiv.org/pdf/2112.06776.pdf"><img src="https://img.shields.io/badge/Paper-PDF-yellowgreen"></a>
  <a href="https://github.com/kgarg8/FullTextKP/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue"></a>
  </a>
</p>

## Create environment
```
conda create -n FKP_env --python==3.6

conda activate FKP_env

conda install pytorch cudatoolkit=11.3 -c pytorch

pip install transformers==4.12.0
```

## Run Commands

## Preprocess
```
cd preprocess

# Stage1
python preprocess_ACM_stage1.py

# Stage2

## Title+Abstract
python preprocess_ACM_stage2_v2.py

## Citations
python preprocess_ACM_stage2_v4.py

## Non-Citations
python preprocess_ACM_stage2_v5.py

## Random
python preprocess_ACM_stage2_v6.py
```

## Summarization
Expects processed_data in the main directory, `pacssum_models` in the summarization folder

Download the pretrained models (into `pacssum_models`) for BERT using `https://drive.google.com/file/d/1wbMlLmnbD_0j7Qs8YY8cSCh935WKKdsP/view?usp=sharing`

```
cd summarization

# Run tfidf summarizer
python run.py --rep tfidf

# Run BERT Summarizer
python run.py --rep bert
```

## Abstractive summarization
```
cd abstractive_summarization

# Stage1
python abs_sum.py

# Stage2
cd preprocess_abs_sum.py

python preprocess_abs_sum.py
```

## Retrieval Augmentation
```
cd specter

python preprocess_ACM.py

./embed.sh
```

## Train & Test
```
# Train
python train.py

# Train on limited data
python train.py --limit=100

# Load Checkpoint
python train.py --checkpoint=True

# Train for multiple runs after the initial run(s)
python train.py --times=3 --initial_time=1

# Test (assuming that saved weights are present)
python train.py --test=True
```

## Citation
Please consider citing our paper if you find this work useful:

```
@article{garg2021keyphrase,
  title={Keyphrase Generation Beyond the Boundaries of Title and Abstract},
  author={Garg, Krishna and Chowdhury, Jishnu Ray and Caragea, Cornelia},
  journal={Findings of EMNLP},
  year={2022}
}
```

## Credits
[PacSum Repo for Summarization](https://github.com/mswellhao/PacSum/tree/637dffeddb0e83a53e73012ca33727c773c2c158) 

[Specter](https://github.com/allenai/specter)

[FAISS](https://github.com/facebookresearch/faiss)

## Questions
Please contact `kgarg8@uic.edu` for any questions related to this work.
