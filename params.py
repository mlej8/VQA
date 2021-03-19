    
# number of ocmputers 
num_nodes = 1

# number of gpus per node 
# NOTE: instead we could use all available gpus by setting gpus=-1 in trainer
num_gpus = 4

# effective batch size is batch_size * gpus * num_nodes, e.g. for batchsize of 32, the effective batch size is 32 * 4 * 1 = 128
batch_size = 32
