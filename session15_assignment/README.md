# ERA1_Session15
## Assignment
    Rewrite the whole code covered in the class in Pytorch-Lightning (code copy will not be provided)
    Train the model for 10 epochs
    Achieve a loss of less than 4

The objective of this assignment:
    Understand the internal structure of transformers, so you can modify at your will. 
    Loss should start from 9-10, and reduce to 4, showing that your code is working

## Solution - Notebook [s15_assignment](https://github.com/sdev2030/ERA1_Transformer_PLUS/blob/main/session15_assignment/s15_assignment.ipynb)
In this we will use Transformer architecture from the class, trained in torch lightning to achieve training accuracy of less than 4 by the 10th epoch. 
Following are the parameters Used for training:
        "batch_size": 6,
        "num_epochs": 10,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",

Following are the model parameter and training loss achieved in training the model for 10 epochs.
- Model Parameters - 75.1M
- Training Loss - 3.67 (Epoch Mean)

Graph showing training loss during training.
![Training Loss Graph](https://github.com/sdev2030/ERA1_Transformer_PLUS/blob/main/session15_assignment/images/training_graph.png)

