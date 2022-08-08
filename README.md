# ImRec:<ins>Im</ins>plicit feedback based <ins>Rec</ins>ommendation framework  

_We opensource the framework and hope it could benefit the community._

## Features

- **Aligning recommendation to Industry**
  - Global time splitting with timestamp (_Same raw source data, same results! Previous random 
    splittings suffer from **data leakage** and do not hold this property._)
  - Strictly predict future interactions.
  
- **Supporting various supervised tasks**
  - Supervised with sampled negatives by a sampling strategy
  - Supervised with all positives and negatives (_All unobserved are negatives_)
  - Self-supervised with observed interactions only

- **Unified and order-invariant grid search (GS) entry**
  - One entry for grid search and per-run of model
  - Reproduce same results no matter what order of hyper-parameters in GS
  - Results are summarized to ease your manual comparison after GS

- Personal mantenance
  - Framework consistence is ensured

## Supported models (_More to expected_):
| **Models**       | **Papers**                                                                                             | **Conference/Journal** | **Code**    |
|------------------|--------------------------------------------------------------------------------------------------------|------------------------|-------------|
| BPR              | BPR: Bayesian Personalized Ranking from Implicit Feedback                                              | UAI'09                 | bpr.py      |
| MultiVAE         | Variational Autoencoders for Collaborative Filtering                                                   | WWW'18                 | multivae.py |
| NGCF             | Neural Graph Collaborative Filtering                                                                   | SIGIR'19               | ngcf.py     |
| EHCF             | Efﬁcient Heterogeneous Collaborative Filtering without Negative Sampling for Recommendation            | AAAI'20                | ehcf.py     |
| BUIR             | Bootstrapping User and Item Representations for One-Class Collaborative Filtering                      | SIGIR'21               | buir_*.py   |
| LR-GCCF          | Revisiting Graph based Collaborative Filtering: A Linear Residual Graph Convolutional Network Approach | AAAI'20                | lrgccf.py   |
| LightGCN         | LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation                        | SIGIR'20               |             |
|                  |                                                                                                        |                        |             |
| **Newly added**  |                                                                                                        |                        |             |
| LayerGCN         | Layer-refined Graph Convolutional Networks for Recommendation                                          | arxiv                  | layergcn.py |
| SelfCF(ed/he/ep) | SelfCF: A Simple Framework for Self-supervised Collaborative Filtering                                 | arxiv                  | selfcf*.py  |


## LayerGCN: Layer-refined Graph Convolutional Networks for Recommendation

<p>
<img src="./images/layergcn.png" width="800">
</p>

## Data  
Download from Google Drive: [Amazon-Vedio-Games/Food etc.](https://drive.google.com/drive/folders/1WqRAeoWWGdZplYkjS4640V7v0urNiTXg?usp=sharing)    

## How to run
`python main.py -m LayerGCN -d food`

You may specify other parameters in CMD or config with `configs/model/*.yaml` and `configs/dataset/*.yaml`.

## Best hyper-parameters for reproducibility
We report the best hyper-parameters of LayerGCN to reproduce the results in Table II of our paper as:  

| Datasets | dropout | reg_weight |
|----------|---------|------------|
| MOOC     | 0.1     | 1e-03      |
| Games    | 0.2     | 1e-03      |
| Food     | 0.1     | 1e-02      |
| Yelp     | 0.2     | 1e-03      |



