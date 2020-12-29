# DrBC in Tensorflow 2.x / Keras
Implementation of DrBC approach in Tensorflow 2.x/Keras.

DrBC is a graph neural network approach to identify high Betweenness Centraliy nodes in a graph

This work is based on the initial DrBC project:
Fan, Changjun and Zeng, Li and Ding, Yuhui and Chen, Muhao and Sun, Yizhou and Liu, Zhong[[Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach]](http://arxiv.org/abs/1905.10418) (CIKM 2019)

Original implementation: https://github.com/FFrankyy/DrBC/
![](https://i.imgur.com/vqBlSUQ.jpg "Demo")


The code folder is organized as follows:
```text
> cpp/                           Contains all the .cpp and .h files
    > PrepareBatchGraph              Prepare the batch graphs used in the tensorflow codes
    > graph                          Basic structure for graphs
    > graphUtil                      Compute the collective influence functions
    > graph_struct                   Linked list data structure for sparse graphs
    > metrics                        Compute the metrics functions such as topk accuracy and kendal tau distance
    > utils                          Compute nodes' betweenness centrality
> drbc/                          Contains all the python files for the training and model definition
> drbcython/                     Contains the python bindings for c++ files defined in 'cpp/'
> experiments/                   Will contain all the experiments in the chronological order (including models and logs)
```


## Prerequisites
Get the source code, and install all the dependencies.
```shell
git clone https://github.com/MartinXPN/DrBC.git
cd DrBC && pip install .
```

## Training
Adjust hyper-parameters in `start.py`, and run the following to train the model
```shell
# Change the hyperparameters in the start.py and then run it
python start.py

# Or alternatively provide all the hyperparameters via command line
python -m drbc.gym --experiment vanilla_drbc - \
        construct_datasets --min_nodes 4000 --max_nodes 5000 --nb_train_graphs 100 --nb_valid_graphs 100 --graphs_per_batch 16 --nb_batches 50 --node_neighbors_aggregation gcn --graph_type powerlaw - \
        construct_model --optimizer adam --aggregation max --combine gru - \ 
        train --epochs 100 --stop_patience 5 --lr_reduce_patience 2

# To see the progress on TensorBoard
tensorboard --logdir experiments/latest/logs

# To see the comparison between all the runs with Aim (you need to have docker running first)
aim up

# Or just view the history logs
cat experiments/latest/logs/history.csv
```


## Reproducing the results in the paper
Download the dataset used for evaluation in the paper available on 
Google Drive ([link](https://drive.google.com/file/d/1nsVX8t5EP3JaTjfeHPf74N21jSDUA8dJ/view?usp=sharing))
or GitHub ([link](https://github.com/MartinXPN/DrBC/releases/download/v0.0.1/datasets.zip)).

Also download the model ([link](https://github.com/MartinXPN/DrBC/releases/download/v0.0.1/best-model.zip)).
Provide the path of the model as `--model_path` in the following step.

To run the evaluation and get the results
```shell
python -m drbc.predict real \
            --model_path experiments/latest/models/best.h5py \
            --data_test datasets/Real/amazon.txt \
            --label_file datasets/Real/amazon_score.txt
```

## Alternatively, to build and run the Dockerfile
```shell
docker build -t drbc .
docker run --gpus all -it --rm -v $(pwd)/experiments:/drbc/experiments -v $(pwd)/datasets:/drbc/datasets -v $(pwd)/.aim:/drbc/.aim drbc
```


## Baselines implementations
| Approach      | Implementation  |
| ------------- | --------------- |
| RK and k-BC   | [https://github.com/ecrc/BeBeCA](https://github.com/ecrc/BeBeCA) |
| KADABRA       | [https://github.com/natema/kadabra](https://github.com/natema/kadabra) |
| ABRA          | Codes in the original paper |
| node2vec      | [https://github.com/snap-stanford/snap/tree/master/examples/node2vec](https://github.com/snap-stanford/snap/tree/master/examples/node2vec) |



## References
To cite the initial work [https://github.com/FFrankyy/DrBC](https://github.com/FFrankyy/DrBC)
```
@inproceedings{fan2019learning,
  title={Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach},
  author={Fan, Changjun and Zeng, Li and Ding, Yuhui and Chen, Muhao and Sun, Yizhou and Liu, Zhong},
  booktitle={Proc. 2019 ACM Int. Conf. on Information and Knowledge Management (CIKM’19)},
  year={2019},
  organization={ACM}
}
```
