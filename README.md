# **DEMETAr: Double Encoder Method for an Equivariant Transformer Architecture**

## **1. Setup**

### **Data**

The data used for this code can either be generated directly or is already included in the code/packages installed.
For QM9, the data is already included by default and is loaded automatically when running the script (after the environment has been setup). Meanwhile, for N-body, run the following script after installing the environment:

```sh

python n_body/dataset/generate_dataset.py --initial_vel 1 --num-train 3000 --length 1000 --length_test 1000 --sufix "small"

```

### **Requirements**

Both a [`.txt`](requirements.txt) and [`.yaml`](environment.yaml) file have been provided for convenience.

1. Create a conda environment and install the packages:
```sh

conda env create -f environment.yaml

```

_Note:_ Be sure to use the enyironment file that corresponds to your device.

2. Activate the environment:
```sh

conda activate egnn-transformer

```

## **2. Training**

To perform the experiments, simply run the following lines:

1. For QM9:
```sh

python main_qm9.py
python transformer_trainer.py

Example notebooks with the runs for our experiments can be found in /notebooks/

```

You can adjust the process using the following parameters:

| **Parameter**  | **Description**                                   | **Default Value**  |
|----------------|---------------------------------------------------|--------------------|
| --epochs       | Number of epochs                                  | 10                 |
| --batch_size   | Batch size                                        | 32                 |
| --num_workers  | Number of workers                                 | 1                  |
| --model_name   | Model                                             | "egnn"             |
| --num_hidden   | Hidden features                                   | 77                 |
| --num_layers   | Number of layers                                  | 7                  |
| --act_fn       | Activation function                               | "silu"             |
| --lr           | Learning rate                                     | 1e-3               |
| --weight_decay | Weight decay                                      | 1e-16              |
| --dataset      | Dataset                                           | "qm9"              |
| --target-name  | Target feature to predict                         | "mu"               |
| --dim          | Dimension                                         | 1                  |
| --seed         | Random seed                                       | 42                 |
| --property     | Label to predict                                  | "homo"             |


2. For N-Body:
```sh

python nbody_egnn_trainer.py
python nbody_transformer_trainer.py

```

You can adjust the process using the following parameters:

| **Parameter**      | **Description**                                   | **Default Value**  |
|--------------------|---------------------------------------------------|--------------------|
| --epochs           | Number of epochs                                  | 100                |
| --batch_size       | Batch size (number of graphs)                     | 100                |
| --lr               | Learning rate                                     | 5e-4               |
| --lr-scheduling    | Use learning rate scheduling (store_true)         | False              |
| --weight_decay     | Weight decay                                      | 1e-8               |
| --val_freq         | Evaluation frequency (number of epochs)           | 10                 |
| --dataset          | Dataset                                           | "charged"          |
| --num_hidden       | Number of values in the hidden layers             | 128                |
| --num_layers       | Number of message passing layers                  | 3                  |
| --basis            | Radial basis function                             | "gaussian"         |
| --double-precision | Use double precision (store_true)                 | False              |
| --model_name       | Model                                             | "egnn"             |
| --seed             | Random seed                                       | 42                 |
| --max_samples      | Maximum number of samples                         | 3000               |


Alternatively, the experiments can be run [here](notebooks/Demo.ipynb).

## **3. References**

Some of the code has been adapted from the following sources:

**N-body:** https://github.com/vgsatorras/egnn

The above can be cited through the following:

```
@article{Satorras2021EnEG,
  title={E(n) Equivariant Graph Neural Networks},
  author={Victor Garcia Satorras and Emiel Hoogeboom and Max Welling},
  journal={ArXiv},
  year={2021},
  volume={abs/2102.09844},
  url={https://api.semanticscholar.org/CorpusID:231979049}
}

```

**Transformer:** https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html
