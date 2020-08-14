"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
from torch.optim import Adam
from torch.optim import SGD
from torch.utils.data import DataLoader
import argparse

from few_shot.datasets import OmniglotDataset, MiniImageNet,CIFARFS
from few_shot.models import get_few_shot_encoder
from few_shot.core import NShotTaskSampler, EvaluateFewShot, create_nshot_task_label,prepare_nshot_task

from few_shot.proto import proto_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH

import numpy as np

np.random.seed(10)
setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True


##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-train', default=1, type=int)
parser.add_argument('--n-test', default=1, type=int)
parser.add_argument('--k-train', default=60, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-train', default=5, type=int)
parser.add_argument('--q-test', default=1, type=int)
parser.add_argument('--meta-batch-size',default=4, type=int)
args = parser.parse_args()

evaluation_episodes = 1000
episodes_per_epoch = 100

if args.dataset == 'omniglot':
    n_epochs = 40
    dataset_class = OmniglotDataset
    num_input_channels = 1
    drop_lr_every = 20
elif args.dataset == 'miniImageNet':
    n_epochs = 100
    dataset_class = MiniImageNet
    num_input_channels = 3
    drop_lr_every = 30
elif args.dataset == 'CIFARFS':
    n_epochs = 100
    dataset_class = CIFARFS
    num_input_channels = 3
    drop_lr_every = 30  
else:
    raise(ValueError, 'Unsupported dataset')

param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
            f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'

print(param_str)

###################
# Create datasets #
###################
background = dataset_class('background')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, episodes_per_epoch, args.n_train, args.k_train, args.q_train,num_tasks=args.meta_batch_size),
    num_workers=8
)
evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test,num_tasks=args.meta_batch_size),
    num_workers=8
)


#########
# Model #
#########
model = get_few_shot_encoder(num_input_channels)
model.to(device, dtype=torch.double)


############
# Training #
############
print(f'Training Prototypical network on {args.dataset}...')
#optimiser = SGD(model.parameters(), lr=1e-3,momentum=0.9,weight_decay=5e-4)
optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss().cuda()


def lr_schedule(epoch, lr):
    # Drop lr every 2000 episodes
    if epoch % drop_lr_every == 0:
        return lr / 2
    else:
        return lr

def prepare_meta_batch(n, k, q, meta_batch_size):
    def prepare_meta_batch_(batch):
        x, y = batch
        # Reshape to `meta_batch_size` number of tasks. Each task contains
        # n*k support samples to train the fast model on and q*k query samples to
        # evaluate the fast model on and generate meta-gradients
        x = x.reshape(meta_batch_size, n*k + q*k, num_input_channels, x.shape[-2], x.shape[-1])
        # Move to device
        x = x.double().to(device)
        # Create label
        y = create_nshot_task_label(k, q).to(device).repeat(meta_batch_size)
        return x, y

    return prepare_meta_batch_

callbacks = [
    EvaluateFewShot(
        eval_fn=proto_net_episode,
        num_tasks=evaluation_episodes,
        n_shot=args.n_test,
        k_way=args.k_test,
        q_queries=args.q_test,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_meta_batch(args.n_test, args.k_test, args.q_test,args.meta_batch_size),
        distance=args.distance
    ),
    ModelCheckpoint(
        filepath=PATH + f'/models/proto_nets/{param_str}.pth',
        monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc'
    ),
    LearningRateScheduler(schedule=lr_schedule),
    CSVLogger(PATH + f'/logs/proto_nets/{param_str}.csv'),
]

fit(
    model,
    optimiser,
    loss_fn,
    epochs=n_epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_meta_batch(args.n_train, args.k_train, args.q_train,args.meta_batch_size),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=proto_net_episode,
    fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                         'distance': args.distance},
)