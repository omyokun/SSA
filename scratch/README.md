
-------------------------

**Improving in-context learning with a better scoring function <br>**


## Environment set up
After cloning the repository, follow the steps below.

1. To install the dependencies, use CONDA. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yml
    conda activate in-context-learning
    ```

2. You can use our trained models in `src\models\polynomial_regression`


3. [Optional] Or you can also train your own models. If you plan to train, populate `conf/wandb.yaml` with you wandb info.


- The `testoverxandy.py`, `testoverlandx.py` `testcross.py` and contains code of tests, you can load our own pre-trained model, you can change parameters and test it for all models
- `train.py` takes as argument a configuration `.yaml` and trains the corresponding model.
- Example command : `python train.py --config src/conf/polynomial_regression.yaml`



