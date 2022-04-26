import argparse
import os
import numpy as np
import copy
from main import main


#####
# maps strings to associated bools
#####
def strtobool(string):
    if string in ["T","t","True","true"]:
        return True
    elif string in ["F","f","False","false"]:
        return False
    else:
        raise Exception(f"String {string} could not be safely converted to bool.")


######
# argument function and constants, must edit together
# available args should always have the names of all arguments that are considered hyperparameters
######
AVAILABLE_ARGS = {'learning_rate':float,'num_steps':int,'anneal_lr':bool,'gae':bool,'gamma':float,
    'gae_lambda':float,'num_minibatches':int,'update_epochs':int,'norm_adv':bool,'clip_coef':float,
    'clip_vloss':bool,'ent_coef':float,'vf_coef':float,'max_grad_norm':float,'target_kl':float}
NUM_HYPERPARAMS = len(AVAILABLE_ARGS)
def parse_args():
    parser = argparse.ArgumentParser()

    # grid search parameters
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="TurtleRLEnv-v0",
        help='the id of the gym environment')
    parser.add_argument('--num-envs', type=int, default=4,
        help='the number of parallel game environments')
    parser.add_argument('--total-timesteps', type=int, default=10_000,
        help='total timesteps of all experiments for a given set of hyperparameters')
    parser.add_argument('--num-explorations', type=int, default=500,
        help='total number of hyper parameter combinations to explore')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='weather to capture videos of the agent performances (check out `videos` folder)')

    # hyper parameters
    parser.add_argument('--learning-rate', type=float, nargs='+', default=[2.5e-4],
        help='the learning rate of the optimizer')
    parser.add_argument('--num-steps', type=int, default=[128], nargs='+',
        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=[True], nargs='+', 
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=[True], nargs='+', 
        help='Use GAE for advantage computation')
    parser.add_argument('--gamma', type=float, default=[0.99], nargs='+',
        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=[0.95], nargs='+',
        help='the lambda for the general advantage estimation')
    parser.add_argument('--num-minibatches', type=int, default=[4], nargs='+',
        help='the number of mini-batches')
    parser.add_argument('--update-epochs', type=int, default=[4], nargs='+',
        help="the K epochs to update the policy")
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=[True], nargs='+', 
        help="Toggles advantages normalization")
    parser.add_argument('--clip-coef', type=float, default=[0.2], nargs='+',
        help="the surrogate clipping coefficient")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=[True], nargs='+', 
        help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')
    parser.add_argument('--ent-coef', type=float, default=[0.01], nargs= '+',
        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=[0.5], nargs='+',
        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=[0.5], nargs='+',
        help='the maximum norm for the gradient clipping')
    parser.add_argument('--target-kl', type=float, default=[None], nargs='+',
        help='the target KL divergence threshold')
    
    args = parser.parse_args()

    #args.batch_size = np.asarray(args.num_envs * np.asarray(args.num_steps), int)
    #num_minibatches = np.array([[1/x for x in args.num_minibatches]])
    #args.minibatch_size = np.asarray(np.outer(args.batch_size, num_minibatches), int)
    #print(args.minibatch_size.shape)

    args.seed = None
    args.torch_deterministic = False
    args.cuda = True
    args.train = True
    args.gui = False

    return args

#######
# select_args converts a dictionary of list-like arguments and a list of integers into
# a list of single-number arguments by randomly selecting from each list
#######
def select_args(args):

    serial = np.zeros(NUM_HYPERPARAMS)
    for i, arg_name in enumerate(AVAILABLE_ARGS.keys()):
        arg_value = getattr(args,arg_name)
        nargs_passed = len(arg_value)

        selected_index = np.random.randint(0,nargs_passed)
        serial[i] = arg_value[selected_index]

    return serial

######
# is_vector_in_matrix checks whether params are not already in history
#
# params is list of hyperparameters of length NUM_HYPERPARAMS
# history is n x NUM_HYPERPARAMS
#
# this function checks each row in history for match with params
######
def is_vector_in_matrix(params, history):
    for i in range(history.shape[0]):
        if np.all(history[i,:] == params):
            return True

    return False

#####
# main function for grid search
#####
if __name__ == "__main__":
    args = parse_args()
    
    # loop through sets of hyperparameters to explore
    params_to_explore = np.zeros([args.num_explorations, NUM_HYPERPARAMS])
    for explore_i in range(args.num_explorations):

        # randomly select one combination of hyperparameters to explore
        params_to_add = np.zeros(NUM_HYPERPARAMS)
        
        while is_vector_in_matrix(params_to_add, params_to_explore):
            params_to_add = select_args(args)

        # add this combo to history
        params_to_explore[explore_i,:] = params_to_add

        # update argument dictionary to reflect selection and execute the trial
        explore_args = copy.deepcopy(args)
        for param_i, param in enumerate(AVAILABLE_ARGS.items()):
            param_name = param[0]
            data_type = param[1]
            setattr(explore_args, param_name, data_type(params_to_add[param_i]))

        print(f"Searching {explore_i+1}/{args.num_explorations}...")
        main(explore_args)