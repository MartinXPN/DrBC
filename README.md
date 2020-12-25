# DrBC++
A graph neural network approach to identify high Betweenness Centraliy in a graph

This work is based on the initial DrBC project:
Fan, Changjun and Zeng, Li and Ding, Yuhui and Chen, Muhao and Sun, Yizhou and Liu, Zhong[[Learning to Identify High Betweenness Centrality Nodes from Scratch: A Novel Graph Neural Network Approach]](http://arxiv.org/abs/1905.10418) (CIKM 2019)


Original implementation: https://github.com/FFrankyy/DrBC/
![](./visualize/Figure_demo.jpg "Demo")

The code folder is organized as follows:
```text
> cpp/                           Contains all the .cpp and .h files
     > PrepareBatchGraph         Prepare the batch graphs used in the tensorflow codes
     > graph                     Basic structure for graphs
     > graphUtil                 Compute the collective influence functions
     > graph_struct              Linked list data structure for sparse graphs
     > metrics                   Compute the metrics functions such as topk accuracy and kendal tau distance
     > utils                     Compute nodes' betweenness centrality
> drbcpp/                        Contains all the python files for the training and model definition
> drbcython/                     Contains the python bindings for c++ files defined in 'cpp/'
> experiments/                   Will contain all the experiments in the chronological order (including models and logs)
```


## Prerequisites
Get the source code, and install all the dependencies.
```shell
git clone https://github.com/MartinXPN/DrBCPP.git
cd DrBCPP && pip install .
```

## Training
Adjust hyper-parameters in `start.py`, and run the following to train the model
```shell
# Change the hyperparameters in the start.py and then run it
python start.py

# Or alternatively provide all the hyperparameters via command line
python -m drbcpp.gym --experiment vanilla_drbc - \
        construct_datasets --min_nodes 4000 --max_nodes 5000 --nb_train_graphs 100 --nb_valid_graphs 100 --graphs_per_batch 16 --nb_batches 50 --node_neighbors_aggregation gcn --graph_type powerlaw - \
        construct_model --optimizer adam --aggregation lstm --combine gru - \ 
        train --epochs 100 --stop_patience 5 --lr_reduce_patience 2
```


## Reproducing the results that reported in the paper
Download the dataset used for evaluation in the paper:
```
https://drive.google.com/file/d/1nsVX8t5EP3JaTjfeHPf74N21jSDUA8dJ/view?usp=sharing 
```

To run the evaluation and get the results
```shell
python predict.py real --model_path experiments/latest/models/best.h5py \
                       --data_test datasets/Real/amazon.txt \
                       --label_file datasets/Real/amazon_score.txt
```

## Alternatively, to build and run the Dockerfile
```shell
docker build -t drbcpp .
docker run --gpus all -it --rm -v $(pwd)/experiments:/drbcpp/experiments -v $(pwd)/datasets:/drbcpp/datasets -v $(pwd)/.aim:/drbcpp/.aim drbcpp
```


## Baselines implementations
| Approach      | Implementation  |
| ------------- | --------------- |
| RK and k-BC   | [https://github.com/ecrc/BeBeCA](https://github.com/ecrc/BeBeCA) |
| KADABRA       | [https://github.com/natema/kadabra](https://github.com/natema/kadabra) |
| ABRA          | Codes in the original paper |
| node2vec      | [https://github.com/snap-stanford/snap/tree/master/examples/node2vec](https://github.com/snap-stanford/snap/tree/master/examples/node2vec) |



## References
To cite our work:
```
TODO
```

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
