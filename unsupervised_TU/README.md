## Preparation 
Please refer to https://chrsmrrs.github.io/datasets/docs/datasets/ to download datasets.
## Training & Evaluation

```
mkdir logs
./run_html.sh $GPU_ID $DATASET_NAME $AUGMENTATION $a $b
```
- `$GPU_ID` is the lanched GPU ID 
- `$DATASET_NAME` is the dataset name. 
- `$AUGMENTATION` could be the following values:
  - `random2` denotes sampling from {NodeDrop, Subgraph}.
  - `random3` denotes sampling from {NodeDrop, Subgraph, EdgePert}.
  - `random4` denotes sampling from {NodeDrop, Subgraph, EdgePert, AttrMask}.
- `$a` is the weight of graph-tier topology isomorphism expertise
- `$b` is the weight of subgraph-tier topology isomorphism expertise












