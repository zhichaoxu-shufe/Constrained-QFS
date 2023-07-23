

## A Lightweight Constrained Generation Alternative for Query-focused Summarization
This is the official code implementation for arXiv manuscript: [A Lightweight Constrained Generation Alternative for Query-focused Summarization](https://dl.acm.org/doi/pdf/10.1145/3539618.3591936)

A significant part of this code repo comes from [Neurological Decoding Code Repo](https://github.com/GXimingLu/neurologic_decoding)

#### Dependency
```bash
pip3 install -r requirements.txt
```

#### Download the processed dataset and neural retrieval model from [Link](https://drive.google.com/file/d/1JQDwl6bEAF_KdhuGqZo1aV0V8YxNP4O-/view?usp=sharing)

#### Finetune the backbone LM
need to configure the path and hyperparameters accordingly, a sample bash file is 
```bash
bash run_finetune.sh
```
#### Decode
need to configure the path and hyperparameters accordingly, a sample bash file is 
```bash
bash run_decode.sh
```


#### Citation:
```bibtex
@inproceedings{xu-cohen-2023-lightweight,
    author = {Xu, Zhichao and Cohen, Daniel},
    title = {A Lightweight Constrained Generation Alternative for Query-Focused Summarization},
    year = {2023},
    isbn = {9781450394086},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3539618.3591936},
    doi = {10.1145/3539618.3591936},
    booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages = {1745–1749},
    numpages = {5},
    keywords = {query-focused summarization, constrained generation},
    location = {Taipei, Taiwan},
    series = {SIGIR '23}
}

@inproceedings{lu-etal-2021-neurologic,
    title = "{N}euro{L}ogic Decoding: (Un)supervised Neural Text Generation with Predicate Logic Constraints",
    author = "Lu, Ximing  and  West, Peter  and  Zellers, Rowan  and  Le Bras, Ronan  and  Bhagavatula, Chandra  and  Choi, Yejin",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.naacl-main.339",
    doi = "10.18653/v1/2021.naacl-main.339",
    pages = "4288--4299",
}

```