import argparse
import itertools
import os
import sys
import time
import pickle
import copy

import dgl
import numpy as np
import torch
# from tqdm import tqdm
import random
sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph
from src.rrgcn import RecurrentRGCN
from src.hyperparameter_range import hp_range
import torch.nn.modules.rnn
import scipy.sparse as sp
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list
# os.environ['KMP_DUPLICATE_LIB_OK']='True'


def print_all_model_parameters(model):
    print('\nModel Parameters')
    print('--------------------------')
    for name, param in model.named_parameters():
        print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))
    param_sizes = [param.numel() for param in model.parameters()]
    print('Total # parameters = {}'.format(sum(param_sizes)))
    print('--------------------------')
    print()


def temporal_regularization(params1, params2):
    regular = 0
    for (param1, param2) in zip(params1, params2):
        regular += torch.norm(param1 - param2, p=2)
    # print(regular)
    return regular
    # param_sizes = [param.numel() for param in model.parameters()]    


def test(model, history_len, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list,
         time_list, model_name, mode):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param mode
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

    idx = 0
    start_time = len(history_list)
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name, checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"start testing"+"-"*10+"\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    # do not have inverse relation in test input
    input_list = [snap for snap in history_list[-history_len:]]

    for time_idx, test_snap in enumerate(test_list):
        tc = start_time + time_idx
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
    
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)

        # get history
        histroy_data = test_triples_input
        inverse_histroy_data = histroy_data[:, [2, 1, 0, 3]]
        inverse_histroy_data[:, 1] = inverse_histroy_data[:, 1] + num_rels
        histroy_data = torch.cat([histroy_data, inverse_histroy_data])
        histroy_data = histroy_data.cpu().numpy()
        all_tail_seq = sp.load_npz(
            '../data/{}/history/tail_history_{}.npz'.format(args.dataset, time_list[time_idx]))
        seq_idx = histroy_data[:, 0] * num_rels * 2 + histroy_data[:, 1]
        tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
        entity_history = tail_seq.masked_fill(tail_seq != 0, 1)
        if use_cuda:
            entity_history = entity_history.cuda()

        final_score, final_r_score = model.predict(history_glist, test_triples_input, use_cuda, entity_history)

        mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(test_triples_input, final_r_score, all_ans_r_list[tc], eval_bz=1000, rel_predict=1)
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples_input, final_score, all_ans_list[tc], eval_bz=1000, rel_predict=0)

        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)

        # relation rank
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)
        mrr_raw_list_r.append(mrr_snap_r)
        mrr_filter_list_r.append(mrr_filter_snap_r)

        # reconstruct history graph list
        input_list.pop(0)
        input_list.append(test_snap)
        idx += 1
    
    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list, train_times = utils.split_by_time_with_times(data.train)
    valid_list, valid_times = utils.split_by_time_with_times(data.valid)
    test_list, test_times = utils.split_by_time_with_times(data.test)
    total_data = np.concatenate((data.train, data.valid, data.test), axis=0)
    print("total data length ", len(total_data))
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    all_ans_list = utils.load_all_answers_for_time_filter(total_data, num_rels, num_nodes, False)
    all_ans_list_r = utils.load_all_answers_for_time_filter(total_data, num_rels, num_nodes, True)

    test_model_name = "{}-{}-ly{}-dilate{}-his{}-dp{}"\
        .format(args.encoder, args.decoder, args.n_layers, args.dilate_len, args.train_history_len, args.dropout)
    if not os.path.exists('../models/{}/'.format(args.dataset)):
        os.makedirs('../models/{}/'.format(args.dataset))
    test_state_file = '../models/{}/{}'.format(args.dataset, test_model_name) 
    print("Sanity Check: stat name : {}".format(test_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    # create stat
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                        num_nodes,
                        num_rels,
                        args.n_hidden,
                        args.opn,
                        sequence_len=args.train_history_len,
                        num_bases=args.n_bases,
                        num_basis=args.n_basis,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        self_loop=args.self_loop,
                        skip_connect=args.skip_connect,
                        layer_norm=args.layer_norm,
                        input_dropout=args.input_dropout,
                        hidden_dropout=args.hidden_dropout,
                        feat_dropout=args.feat_dropout,
                        entity_prediction=args.entity_prediction,
                        relation_prediction=args.relation_prediction,
                        use_cuda=use_cuda,
                        gpu = args.gpu,
                        global_rate = args.global_rate)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    if args.test == -1:
        print("----------------------------------------start training model with history length {}----------------------------------------\n".format(args.train_history_len))
        model_name = "{}-{}-ly{}-dilate{}-his{}-dp{}"\
            .format(args.encoder, args.decoder, args.n_layers, args.dilate_len, args.train_history_len, args.dropout)
        if not os.path.exists('../models/{}/'.format(args.dataset)):
            os.makedirs('../models/{}/'.format(args.dataset))
        model_state_file = '../models/{}/{}'.format(args.dataset, model_name) 
        print("Sanity Check: stat name : {}".format(model_state_file))
        print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))
            
        best_mrr = 0
        best_epoch = 0
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            # losses_e = []
            # losses_r = []

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)
            for train_sample_num in idx:
                if train_sample_num == 0 or train_sample_num == 1: continue
                if train_sample_num - args.train_history_len<0:
                    input_list = train_list[0: train_sample_num]
                    output = train_list[1:train_sample_num+1]
                else:
                    input_list = train_list[train_sample_num-args.train_history_len: train_sample_num]
                    output = train_list[train_sample_num-args.train_history_len+1:train_sample_num+1]

                # generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [torch.from_numpy(_).long() for _ in output]

                histroy_data = output[-1]
                inverse_histroy_data = histroy_data[:, [2, 1, 0, 3]]
                inverse_histroy_data[:, 1] = inverse_histroy_data[:, 1] + num_rels
                histroy_data = torch.cat([histroy_data, inverse_histroy_data])
                histroy_data = histroy_data.cpu().numpy()
                # tail
                all_tail_seq = sp.load_npz(
                    '../data/{}/history/tail_history_{}.npz'.format(args.dataset, train_times[train_sample_num]))
                seq_idx = histroy_data[:, 0] * num_rels * 2 + histroy_data[:, 1]
                tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
                entity_history = tail_seq.masked_fill(tail_seq != 0, 1)
                if use_cuda:
                    entity_history = entity_history.cuda()

                loss = model.get_loss(history_glist, output[-1], None, use_cuda, entity_history)

                # loss = args.task_weight * loss_e + (1 - args.task_weight) * loss_r

                losses.append(loss.item())
                # losses_e.append(loss_e.item())
                # losses_r.append(loss_r.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()

            print("His {:04d}, Epoch {:04d} | Ave Loss: {:.4f} | Best MRR {:.4f} | Model {} "
                .format(args.train_history_len, epoch, np.mean(losses), best_mrr, model_name))

            # validation        
            if epoch % args.evaluate_every == 0:
                mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
                                                                    args.train_history_len, 
                                                                    train_list, 
                                                                    valid_list, 
                                                                    num_rels, 
                                                                    num_nodes, 
                                                                    use_cuda, 
                                                                    all_ans_list, 
                                                                    all_ans_list_r,
                                                                    valid_times,
                                                                    model_state_file, 
                                                                    mode="train")
                
                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_filter< best_mrr:
                        if epoch >= args.n_epochs or epoch - best_epoch > 5:
                            break
                    else:
                        best_mrr = mrr_filter
                        best_epoch = epoch
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                else:
                    if mrr_filter_r < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_filter_r
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                            args.train_history_len, 
                                                            train_list+valid_list,
                                                            test_list, 
                                                            num_rels, 
                                                            num_nodes, 
                                                            use_cuda, 
                                                            all_ans_list, 
                                                            all_ans_list_r,
                                                            test_times,
                                                            model_state_file, 
                                                            mode="test")
        return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    elif args.test == 0:
        # load best model with start history length
        init_state_file = '../models/{}/'.format(args.dataset) + "{}-{}-ly{}-dilate{}-his{}-dp{}"\
            .format(args.encoder, args.decoder, args.n_layers, args.dilate_len, args.train_history_len, args.dropout)
        init_checkpoint = torch.load(init_state_file, map_location=torch.device(args.gpu))
        print("Load Previous Model name: {}. Using best epoch : {}".format(init_state_file, init_checkpoint['epoch']))  # use best stat checkpoint
        print("\n"+"-"*10+"Load model with history length {}".format(args.train_history_len)+"-"*10+"\n")
        model.load_state_dict(init_checkpoint['state_dict'])
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model, 
                                                            args.train_history_len,
                                                            train_list+valid_list,
                                                            test_list, 
                                                            num_rels, 
                                                            num_nodes, 
                                                            use_cuda, 
                                                            all_ans_list, 
                                                            all_ans_list_r,
                                                            test_times,
                                                            init_state_file, 
                                                            mode="test")
        return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MGHGN')

    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--test", type=int, default=0,
                        help="1: formal test 2: continual test")
  
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")

    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    
    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=1,
                        help="weight of entity prediction task")
    parser.add_argument("--kl-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
   
    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")

    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=False,
                        help="add entity prediction loss")


    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--ft_epochs", type=int, default=30,
                        help="number of minimum fine-tuning epoch")
    parser.add_argument("--ft_lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--norm_weight", type=float, default=0.1,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=20,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")

    parser.add_argument("--alpha", type=float, default=0.5,
                        help="hyprparameter of balance encoders")
    
    parser.add_argument("--beta", type=float, default=0.5,
                    help="hyprparameter of balance losses")



    args = parser.parse_args()
    print(args)
    if args.grid_search:
        out_log = '{}.{}.gs'.format(args.dataset, args.encoder+"-"+args.decoder)
        o_f = open(out_log, 'w')
        print("** Grid Search **")
        o_f.write("** Grid Search **\n")
        hyperparameters = args.tune.split(',')

        if args.tune == '' or len(hyperparameters) < 1:
            print("No hyperparameter specified.")
            sys.exit(0)
        grid = hp_range[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range[hp])
        hits_at_1s = {}
        hits_at_10s = {}
        mrrs = {}
        grid = list(grid)
        print('* {} hyperparameter combinations to try'.format(len(grid)))
        o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
        o_f.close()

        for i, grid_entry in enumerate(list(grid)):

            o_f = open(out_log, 'a')

            if not (type(grid_entry) is list or type(grid_entry) is list):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)
            print('* Hyperparameter Set {}:'.format(i))
            o_f.write('* Hyperparameter Set {}:\n'.format(i))
            signature = ''
            print(grid_entry)
            o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")
            # def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
            mrr, hits, ranks = run_experiment(args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3])
            print("MRR (raw): {:.6f}".format(mrr))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr))
            for hit in hits:
                avg_count = torch.mean((ranks <= hit).float())
                print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, avg_count.item()))
    # single run
    else:
        run_experiment(args)
    sys.exit()



