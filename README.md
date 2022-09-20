# ProBindNN

## General use
The code used in my bachelor thesis on predicting mutation effects on protein-protein binding affinities using graph neural networks.
The development of the took place on Linux in a Conda environment. To install all the dependencies run the following code in virtual environment in the terminal:

```
 python -m pip install -r requirements.txt
```
With the requirements.txt provided in the repository.

The usage of the aboe code is as follows:
For the creation of the dataset make\_dataset.py can be used either as an import into a submodule, or directly calling

 ```
 cd ProBindNN
 nohup python src/make_dataset.py > logs/nohup/make_dataset.txt
 ```
 
 The folder of the dataset will be automatically created.
 For training the same applies, one can either import the train method, or directly run it from the command line:
 
 ```
 nohup python src/make_dataset.py > training.txt
 ```

The use of nohup is just an optional convenience for logging purposes.
The automatically generated logs folder contains a tensorboard subfolder, where the training and validation losses are saved with a unique timestamp as an identifier. To look at the data, use the following command:

```
tensorboard --logdir logs/tensorboard
```
This will automatically launch Tensorboard and open a browser window where the training process can be observed. 

Another option is to open the main.ipynb Jupiter notebook and run it line by line, remember this is only recommended for short sessions, because the python kernel dies after closing your IDE.
## Plans for the future
A useful upgrade in the future would be to use a hyperparameter tuner like RayTune.
Also, graphein is a really nice package, but the edge construction functions have to be costumized more.

My thesis can be found [here](./_paper/Thesis_Babai.pdf).

If you have some issues with the code or just want to share your opinion on my work, don't hesitate to contact me!
