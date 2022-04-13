# Bayesian Framework for Gradient Leakage <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="Secure, Reliable, and Intelligent Systems Lab" align="right" src="https://www.sri.inf.ethz.ch/assets/images/sri-logo.svg"></a>

This it the official codebase for the paper [Bayesian Framework for Gradient Leakage (ICLR 2022)](https://www.sri.inf.ethz.ch/publications/balunovic2022bayesian).


In the paper, we developed a theoretical framework that formalizes and generalizes recent attacks that reveal private information from the gradient updates shared during federated learning.
The theoretical framework introduces the idea of Bayes optimal adversary that is phrased as an optimization problem. The problem is shown to be solvable if the defense and data distributions are known. 
In our paper, we further developed approximations to the Bayes optimal adversary and show experimentally that they are effective.
In this repository, we provide the implementation of our comparison experiments, which demonstrate that approximations of the Bayes optimal adversary for a known defense mechanism are superior to other attack strategies. 
Further, we provide the implementation of our experiments that break existing defenses, demonstrated in Figure 2 of our paper.

## Getting started

Install [Anaconda](https://conda.io/) and execute the following command:
```
conda env create --name federated --file environment.yml
conda activate federated_bayes
```

To download the pretrained models for all experiments below execute the following command:
```
./download_models.sh
```

## Comparison between the L<sub>1</sub>, L<sub>2</sub>, cos and our Bayes optimal attack

The code for comparing our Bayes optimal attack against L<sub>1</sub>, L<sub>2</sub> and cos attacks is contained in the ***bayes\_optimal\_attack*** folder.

### Running the attacks

To reproduce the results in the experiment of Table 2 in our paper, you can run:
```
./scripts/run_mnist_tune.sh {path_to_model} {defense_name} {output_file}
```
for MNIST results or you can run:

```
./scripts/run_cifar_tune.sh {path_to_model} {defense_name} {output_file}
```
for CIFAR-10 results.

Here ```path_to_model``` is path to one of the models in the ***trained\_models*** folder, ```defense_name``` denotes defense
used to train the model, which must be one of the following: ```dp_gaussian_0.1```, ```dp_laplacian_0.1```, ```soft_gaussian_pruning_0.5_0.1```, ```soft_pruning_0.5_0.1```. 
Finally, ```output_file``` is the name of the file where the results of the experiment will be stored in after running the script.

The get the results for our experimens with stronger priors in Table 5 can be reproduced by running:
```
./scripts/run_mnist_tune_l2_prior.sh {path_to_model} {defense_name} {output_file}
```

## Attack on ATS

The code for our attack on the [ATS defense](https://arxiv.org/abs/2011.12505) is contained in the ***defense\_attacks/ats*** folder.
This part of the repository builds directly on top of the [ATS repository](https://github.com/gaow0007/ATSPrivacy).

### Running the attack

We provide two scripts to train and attack the networks: 
```
benchmark/run/benchmark_attack_step.sh
``` 
Trains and afterwards attacks the network after k steps (batches). Here the number of steps should be greater than 0.
```
benchmark/run/benchmark_attack.sh
``` 
Trains and afterwards attacks the network after k epochs.

In case you want to run the python script directly, the appropriate usage is (for common variables refer to the scripts):

```
python3 -u benchmark/cifar_attack.py --data=$data --arch=$arch --epochs=$epoch_ --aug_list=$aug_list --mode=aug --optim='inversed'
```

For a usage with steps refer to the ```benchmark/run/benchmark_attack_step.sh``` script.
Both scripts train the networks initially. Once you have the networks already trained, you may comment the respective lines in the scripts to avoid unnecessary re-training. As we provide several checkpoints already in ```checkpoints``` the training is by default commented out.

For an overview of all supported models refer to the ```inversefed/nn/models.py``` file, specifically the ```construct_model``` function

### Getting statistics

We provide a python script ```vis.py``` which goes over the resulting folders and aggregates their information.
Simply run it after setting the corresponding fields marked in the script. Before running ```vis.py``` make sure that the folder contains the required ```metrics.npy``` file which is added at the end of an attack.

## Attack on Soteria

The code for our attack on the [Soteria defense](https://arxiv.org/pdf/2012.06043.pdf) is contained in the ***defense_attacks/soteria*** folder.
This part of the repository builds directly on top of the [Soteria repository](https://github.com/jeremy313/Soteria).

### Models

We provide our pre-trained models in the ***models*** subdirectory. All our models have been trained with the [Inverting Gradients](https://github.com/JonasGeiping/invertinggradients) library, the adapted code for which (for step-wise training) can be found in the repository of our attack on ATS.

### Running the attack

We provide one script and one python file to run our attack: 
Via ```reconstruct_image_set.py``` you can attack a network by running (refer to the ```.sh``` scripts for values):

```
python3 -u reconstruct_image_set.py --model=$arch --defense=ours --pruning_rate=$pruning_rate --tv=$tv --save_imag --steps=$ctr --use_steps
```

If you use the ```--use_steps``` flag we search for a model trained for ```$ctr``` steps. Otherwise the ```--steps``` flag refers to the epochs of the model.

Via ```run_trained_experiments.py``` you can run mutliple experiments as above using a single script.

In order to not affect the API of the inversefed library we have to switch to our attack manually (enabled by default). In particular one should follow the steps below to defend / attack a specific layer in a network.


1. Choose a model which is  ```inversefed/nn/models.py``` or insert your new model there. We work on the already included ConvBig model.
2. Adapt your selected model in ```inversefed/nn/models.py``` by letting the layer before the defended layer ouput to self.feature (See our ConvBig implementation)
3. Change the index in line 132 ```reconstruct_image_set.py``` to represent the defended layer's weights. Note that -1 = last layer bias, -2 = last layer weights, -3 = second to last layer bias ...
4. If you want to use our attack on layer k do the following
    - Go to ```inversefed/reconstruction_algorithms.py``` Line 372 (Marked)
    - Set remove_defense to True
    - Set attack_idx to k

Note that we do not train the network here (we provide trained checkpoints). If you want to train your own network we refer to our attack on ATS (or simply the inversefed library).

### Getting statistics

We provide a python script ```vis.py``` which goes over the resulting folders and aggregates their information.
Simply run it after setting the corresponding fields marked in the script. Before running ```vis.py``` make sure that the folder contains the required ```metrics.npy``` file which is added at the end of an attack.

## Attack on PRECODE

The code for our attack on the [PRECODE defense](https://openaccess.thecvf.com/content/WACV2022/papers/Scheliga_PRECODE_-_A_Generic_Model_Extension_To_Prevent_Deep_Gradient_WACV_2022_paper.pdf) is contained in the ***defense\_attacks/precode*** folder.
We note that the ***inversefed*** directory supplied was originally taken from the ['Inverting Gradients - How easy is it to break Privacy in Federated Learning?'](https://github.com/JonasGeiping/invertinggradients) paper and later modified by us for the purposes of this paper. 

### Running the attack


To execute an attack on the PRECODE model used in the paper run the command:
```
python3 reconstruct.py -m ./models/CIFAR10_ffn_ReLU_500.ckpt
```

A figure with resulting reconsructions is generated in the same directory, called ***precode.pdf*** .

For more attack options execute:
```
 python3 reconstruct.py --help
```
### Pretrained models 


Further we supply additional versions of the same model at different training steps in the ***models*** directory.

### Training 


To train a PRECODE model from scratch you can do run the command:
```
python3 train.py -i 5000
```
The resulting models will appear in in the ***models*** directory.
For more training options execute:
```
python3 train.py --help
```

## Contributors

- [Mislav Balunović](https://www.sri.inf.ethz.ch/people/mislav)
- [Dimitar I. Dimitrov](https://www.sri.inf.ethz.ch/people/dimitadi)
- Robin Staab
- [Martin Vechev](https://www.sri.inf.ethz.ch/people/martin)

## Citation

If you find this work useful for your research, please cite it as:

```
@inproceedings{
	balunovic2022bayesian,
	title={Bayesian Framework for Gradient Leakage},
	author={Mislav Balunovic and Dimitar Iliev Dimitrov and Robin Staab and Martin Vechev},
	booktitle={International Conference on Learning Representations},
	year={2022},
	url={https://openreview.net/forum?id=f2lrIbGx3x7}
}
```




