This code is for `Re-examining learning linear functions in context` experiments, we based ourselves on the [code](https://github.com/dtsip/in-context-learning) of Garg et al. for their paper *[What Can Transformers Learn In-Context? A Case Study of Simple Function Classes](http://arxiv.org/abs/2208.01066)*. The included LICENSE file is from their original code and is retained to comply with the terms of the MIT License, which requires preservation of the original copyright notice and permission notice, even in modified versions.


-------------------------
**Re-examining learning linear functions in context** <br>


## Getting started
Please start by cloning our repository and follow the steps below.

1. To install the dependencies, use CONDA. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yml
    conda activate in-context-learning
    ```

2. You can use our trained models in `src\models\polynomial_regression`


3. [Optional] Or you can also train your own models. If you plan to train, populate `conf/wandb.yaml` with you wandb info.


- The `eval.ipynb` notebook contains code to load our own pre-trained model, you can change parameters and test it for all models
- `train.py` takes as argument a configuration `.yaml` and trains the corresponding model.
- Example command : `python train.py --config src/conf/polynomial_regression.yaml`




