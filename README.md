# R-ESM
Evolutionary Scale Modeling (ESM) for RNA

Built based on [ESM](https://github.com/facebookresearch/esm) (A Transformer protein language models from Facebook AI Research).

## Usage
```shell
pip install torch
pip install fair-esm
./exp.sh 
```
## Features
Training script for ESM.

RNA codens as tokens.

A pre-trained model (35M) on snRNA dataset:
| Scale | Coden Size | Number of layers | Embedding dim | Attention heads|
|:----:|:----:|:----:|:----:|:----:|
| 35M | 3 | 12 | 480 | 20 |

<img src=asset/intro.png width=300 height=400 />
<img src=asset/downstream.png width=300 height=400 />

