# [Evaluating-Dynamic-Topic-Models](https://aclanthology.org/2024.acl-long.11.pdf)
A DTM captures the evolution of topics over time in a corpus. This paper proposes new ways to evaluate that evolution by quantifying topic quality and consistency across timestamps.

## Installation
Create & activate Conda env

```
conda env create -f environment.yml
conda activate DTQ
```

## Running the Code

```
python main.py
```

## Citation
If you find this helpful, feel free to cite the following papers.

```bibtex
@inproceedings{karakkaparambil-james-etal-2024-evaluating,
    title = "Evaluating Dynamic Topic Models",
    author = "Karakkaparambil James, Charu  and
      Nagda, Mayank  and
      Haji Ghassemi, Nooshin  and
      Kloft, Marius  and
      Fellenz, Sophie",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.11/",
    doi = "10.18653/v1/2024.acl-long.11",
    pages = "160--176",
    abstract = "There is a lack of quantitative measures to evaluate the progression of topics through time in dynamic topic models (DTMs). Filling this gap, we propose a novel evaluation measure for DTMs that analyzes the changes in the quality of each topic over time. Additionally, we propose an extension combining topic quality with the model{'}s temporal consistency. We demonstrate the utility of the proposed measure by applying it to synthetic data and data from existing DTMs, including DTMs from large language models (LLMs). We also show that the proposed measure correlates well with human judgment. Our findings may help in identifying changing topics, evaluating different DTMs and LLMs, and guiding future research in this area."
}
```
