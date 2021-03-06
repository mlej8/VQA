# parameters for general setting
savepath = 'model'

# parameters for the visual feature
vfeat = 'googlenetFC'
vdim = 1024

# parameters for data pre-process
thresh_questionword = 6
thresh_answerword = 3
batch_size = 32
seq_length = 50

# parameters for learning
uniformLR = 0
epochs = 50
nepoch_lr = 100
decay = 1.2
embed_word = 1024

# parameters for universal learning rate
maxgradnorm = 20
maxweightnorm = 2000

# parameters for different learning rates for different layers
lr_wordembed = 0.8
lr_other = 0.01
lr = 1e-3
weightClip_wordembed = 1500
weightClip_other = 20

# dataloader settings
num_workers = 8
shuffle = True

# optimal number of epochs based on hyperparameter tuning on val dataset
opt_epochs = 15