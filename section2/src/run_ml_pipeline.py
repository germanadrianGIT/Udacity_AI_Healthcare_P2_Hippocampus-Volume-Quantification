"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import numpy as np
from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = "../out/"
        # self.n_epochs = 15
        self.n_epochs = 1
        self.learning_rate = 0.00002
        # self.batch_size = 8
        self.batch_size = 2
        self.patch_size = 64
        self.test_results_dir = "../results"

if __name__ == "__main__":
    # Get configuration

    # TASK: Fill in parameters of the Config class and specify directory where the data is stored and 
    # directory where results will go
    c = Config()

    # Load data
    print("Loading data...")

    # TASK: LoadHippocampusData is not complete. Go to the implementation and complete it. 
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)


    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality

    keys = range(len(data))

    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 

    split = dict()

    # TASK: create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    # <YOUR CODE GOES HERE>
    
    # Randomly select 80% of the dataset indices for training
    train = np.random.choice(keys, size=int(0.8 * len(data)), replace=False)

    # Assign the remaining 20% of the indices to testing/validation candidates
    test = [i for i in keys if i not in train]

    # Split the remaining 20% in half for validation
    val = test[:len(test) // 2]

    # And use the other half as the final test set
    test = test[(len(test) // 2):]
    
    split["train"] = train
    split["val"] =val
    split["test"] =test
    print ("Number of files in train, validation and test set are :  ",len(train), len(val),len(test))
    # Set up and run experiment
    
    # TASK: Class UNetExperiment has missing pieces. Go to the file and fill them in
    print("SPLIT STRUCTURE:")
    for k, v in split.items():
        print(f"{k}: type={type(v)}, len={len(v)}")

    exp = UNetExperiment(c, split, data)

    # You could free up memory by deleting the dataset
    # as it has been copied into loaders
    # del dataset 
    del data
    # run training
    exp.run()

    # prep and run testing

    # TASK: Test method is not complete. Go to the method and complete it
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))