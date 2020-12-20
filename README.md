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


## To run the project
Get the source code, and install all the dependencies.
```
git clone https://github.com/MartinXPN/DrBCPP.git
cd DrBCPP && pip install .
```

## Training
Adjust hyper-parameters in `start.py`, and run the following to train the model
```
# Change the hyperparameters in the start.py and then run it
python start.py

# Or alternatively provide all the hyperparameters via command line
python -m drbcpp.gym --experiment my_experiment_name - \
        construct_datasets --min_nodes 2 --max_nodes 1000 --nb_train_graphs 123 --nb_valid_graphs 123 --graphs_per_batch 20 --nb_batches 20 - \
        construct_model - \ 
        train --epochs 10
```


## Reproducing the results that reported in the paper
Here is the link to the dataset that was used in the paper:
```
TODO
```

To run the evaluation and get the results
```shell script
predict.py real --model_path experiments/<DATE>/models/best.h5py --data_test Data/Real/amazon.txt --label_file Data/Real/exact_bc/amazon.txt
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
