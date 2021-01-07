# CS867 Assignment 3: Exploring CNN with KERAS
The main purpose of this assignmnet is to explore KERAS by implementing basic neural network and understand diverse tools offered by KERAS.
Here are list of things that exists in this repository.
- In **python_files** folder **.py** files (In each file the exist a arthitecture that is implemented and code is runable)
- In **graphs** folder images of graphs and results which includes (Stats of lose function, Confusion Matrices  etc.)
- In **model_weights** folder there exist pretrained model weights stored as instructed
- In **jupyter_notebooks** folder the already implemented expamples on colab are shared 
>It is assumed in given code that dataset is downloaded for implementations and stored in seprate folders in form of categories.
# Running the given code
1. The evaluate has to place code files and dataset folders in same folder or same level of hierarchy.
2. Provide the path to dataset folder in veriable of **path_train, path_test, and path_pred** than proceed with running code.
# Details regarding task
These are the architecture implemnted in this task with the architecture details. Here as it is obserable Case 3 is more successful.
|Case Number| Architecture  | Number of epochs | Loss | Accuracy |
|-------| ------------- | ---------------- | ---- | -------- |
|1| VGG Trasnfer Learning (flatten, FC RELU, Softmax)  | 10  | 0.2735 | 0.9077 |
|2| VGG Trasnfer Learning (flatten, FC RELU, Softmax, layer 16 is Frozen)  | 10  | 0.2565 | 0.9083 |
|3| **VGG Trasnfer Learning (flatten, FC RELU, Softmax)**  | **10**  | **0.2306** | **0.9147** |
# Learning curve
![Loss and Accuracy per Epoch Case 1.png](graphs/Loss%20and%20Accuracy%20per%20Epoch%20Case%201.png)
![Loss and Accuracy per Epoch Case 2.png](graphs/Loss%20and%20Accuracy%20per%20Epoch%20Case%202.png)
![Loss and Accuracy per Epoch Case 3.png](graphs/Loss%20and%20Accuracy%20per%20Epoch%20Case%203.png)
# Confusion Matrices
## Case 1
![Confusion Matrix Case 1.png](graphs/Confusion%20Matrix%20Case%201.png)
## Case 2
![Confusion Matrix Case 2.png](graphs/Confusion%20Matrix%20Case%202.png)
## Case 3
![Confusion Matrix Case 3.png](graphs/Confusion%20Matrix%20Case%203.png)
# Network Diagram
![Network Diagram.png](graphs/Network%20Diagram.png)
