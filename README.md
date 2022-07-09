# ImRec:**Im**plicit feedback based **Rec**ommendation framework
_We opensource the framework and wish it could benefit the community._

## Features
- **Supporting various supervised tasks**
  - Supervised with sampled negatives by a sampling strategy
  - Supervised with all positives and negatives (_All unobserved are negatives_)
  - Self-supervised with observed interactions only

- **Unified and order-invariant grid search (GS) entry**
  - One entry for grid search and per-train of model
  - Reproduce same results no matter what order of hyper-parameters in GS
  - Results summarized to ease your manual comparison after GS

- **Align recommendation with Industry**
  - Global time splitting with timeline (_Same source data, same results! Previous random 
    splittings suffer from **data leakage** and do not hold this property._)
  - Strictly predict future interactions.

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



