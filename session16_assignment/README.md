# ERA1_Session16
## Assignment
    1. Pick the "en-fr" dataset from opus_books 
    2. Remove all English sentences with more than 150 "tokens" 
    3. Remove all french sentences where len(fench_sentences) > len(english_sentrnce) + 10 
    4. Train your own transformer (E-D) (do anything you want, use PyTorch, OCP, PS, AMP, etc), but get your loss under 1.8

## Solution - Notebook [s16_assignment](https://github.com/sdev2030/ERA1_Transformer_PLUS/blob/main/session16_assignment/s16_assignment.ipynb)
In this we will use Transformer architecture from the class, trained in torch lightning to achieve training accuracy of less than 1.8 . 
We have implemented:
 - dynamic padding (mini-batch level padding) using collate_fn of dataloader 
 - One Cycle Policy in lightning optimizer setup 
 - mixed-precision with 'fp16' in ligthning trainer and 
 - parameter sharing in function build_transformer.

Following are the parameters Used for training:

        "batch_size": 8,
        "num_epochs": 10,
        "lr": 10**-4,
        "seq_len": 160,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "fr",

Following are the model parameter and training loss achieved in training the model for 10 epochs.
- Model Parameters - 68.1M
- Training Loss - 1.452 (Epoch Mean)

Graph showing training loss during training.
![Training Loss Graph](https://github.com/sdev2030/ERA1_Transformer_PLUS/blob/main/session16_assignment/images/training_graph.png)

Graph showing the LR used by One Cycle policy.
![One Cycle Policy LR Graph](https://github.com/sdev2030/ERA1_Transformer_PLUS/blob/main/session16_assignment/images/lr_graph.png) 

