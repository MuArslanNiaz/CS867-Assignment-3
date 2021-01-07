# CS867 Assignment 3: Exploring CNN with KERAS
The main purpose of this assignmnet is to explore KERAS by implementing basic neural network and understand diverse tools offered by KERAS.
Here are list of things that exists in this repository.
- In **Code** folder **.py** files (In each file the exist a arthitecture that is implemented and code is runable)
- In **Graphs** folder images of graphs and results which includes (Stats of lose function, Confusion Matrices  etc.)
- In **Models** folder there exist pretrained model weights stored as instructed
- In **Notebooks** folder the already implemented expamples on colab are shared 
>It is assumed in given code that dataset is downloaded for implementations and stored in seprate folders in form of categories.
# Running the given code
1. The evaluate has to place code files and dataset folders in same folder or same level of hierarchy.
2. Provide the path to dataset folder in veriable of **path_train, path_test, and path_pred** than proceed with running code.
# Details regarding task
These are the architecture implemnted in this task with the architecture details.
| Atchitecture  | Number of epochs | Loss | Accuracy |
| ------------- | ---------------- | ---- | -------- |
| Basic VGG     | 30               |      |          |
| VGG Trasnfer Learning (flatten, FC RELU, Softmax)  | 50  | 0.23089 | 0.91900 |
