import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from src.model import BaseRGCN
from src.decoder import TConvTransE

class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            for i, layer in enumerate(self.layers):
                layer(g, [])
            return g.ndata.pop('h')
        else:
            raise NotImplementedError

class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, alpha=0.5, beta=0.5):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.h_o = None
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.gpu = gpu

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        self.loss_e = torch.nn.CrossEntropyLoss()


        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        self.tcn_weight_list = nn.ParameterList()
        self.tcn_bias_list = nn.ParameterList()

        for _ in range(sequence_len):
            self.tcn_weight_list.append(nn.Parameter(torch.Tensor(h_dim, h_dim)))
            self.tcn_bias_list.append(nn.Parameter(torch.Tensor(h_dim)))
            nn.init.xavier_uniform_(self.tcn_weight_list[-1], gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.tcn_bias_list[-1])
            
        self.alpha = alpha
        self.beta = beta

        self.relation_cell_1 = nn.GRUCell(self.h_dim * 2, self.h_dim)

        if decoder_name == "tconvtranse":
            self.decoder_ob1 = TConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout,
                                         sequence_len=self.sequence_len)
            self.decoder_ob2 = TConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout,
                                          sequence_len=self.sequence_len)
        else:
            raise NotImplementedError

    def global_forward(self, g_list, use_cuda):
        evolve_embs = []
        self.h_o = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb
        for i, g in enumerate(g_list):
            g = g.to(self.gpu)

            current_h = self.rgcn.forward(g, self.h_o)
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            # time_weight = F.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            # self.h = time_weight * current_h + (1 - time_weight) * self.h
            # # self.h = self.entity_cell_1(current_h, self.h)
            # self.h = F.normalize(self.h)
            evolve_embs.append(current_h)
            # evolve_rels.append(self.h_0)
            # evolve_embs.append(F.normalize(self.rgcn.forward(g, self.h)))
        return self.get_tcn_embeddings(evolve_embs)

    def get_tcn_embeddings(self, embs_list: list, mode='e') -> list:
        res_list = [embs_list[-1]]
        n = len(embs_list)
        if mode == 'e':
            for i in range(n-1):
                tmp_list = []
                m = len(embs_list)-1
                for j in range(m):
                    time_weight = F.sigmoid(torch.mm(embs_list[j], self.tcn_weight_list[n-2-i]) + self.tcn_bias_list[m-1-j])

                    tmp_list.append(F.normalize(time_weight * embs_list[j] + (1 - time_weight) * embs_list[j+1]))
                res_list.append(tmp_list[-1])
                embs_list = tmp_list

        return res_list


    def forward(self, g_list, use_cuda):
        evolve_embs = []
        # evolve_rels = []
        self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb
        for i, g in enumerate(g_list):
            g = g.to(self.gpu)

            temp_e = self.h[g.r_to_e]
            x_input = torch.zeros(self.num_rels * 2, self.h_dim).float().cuda() if use_cuda else torch.zeros(
                self.num_rels * 2, self.h_dim).float()
            for span, r_idx in zip(g.r_len, g.uniq_r):
                x = temp_e[span[0]:span[1], :]
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_input[r_idx] = x_mean
            if i == 0:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.emb_rel)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            else:
                x_input = torch.cat((self.emb_rel, x_input), dim=1)
                self.h_0 = self.relation_cell_1(x_input, self.h_0)
                self.h_0 = F.normalize(self.h_0) if self.layer_norm else self.h_0
            
            current_h = self.rgcn.forward(g, self.h)
            current_h = F.normalize(current_h) if self.layer_norm else current_h
            time_weight = F.sigmoid(torch.mm(self.h, self.time_gate_weight) + self.time_gate_bias)
            self.h = time_weight * current_h + (1-time_weight) * self.h
            # self.h = self.entity_cell_1(current_h, self.h)
            self.h = F.normalize(self.h)
            evolve_embs.append(self.h)
            # evolve_rels.append(self.h_0)
            # evolve_embs.append(F.normalize(self.rgcn.forward(g, self.h)))
        return evolve_embs, self.h_0

    def predict(self, test_graph, test_triplets, use_cuda, entity_global_history=None):
        with torch.no_grad():
            inverse_test_triplets = test_triplets[:, [2, 1, 0, 3]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + self.num_rels
            all_triples = torch.cat((test_triplets, inverse_test_triplets))

            # scores = torch.zeros(len(all_triples), self.num_ents).cuda()
            evolve_embeddings = []
            # evolve_embeddings_list = []
            evolve_rels_list = []
            for idx in range(len(test_graph)):
                # model_idx = len(test_graph) - idx - 1
                evolve_embs, evolve_rels = self.forward(test_graph[idx:], use_cuda)
                # evolve_embeddings_list.append(evolve_embs)
                evolve_rels_list.append(evolve_rels)
                evolve_embeddings.append(evolve_embs[-1])
            # evolve_embeddings = self.get_tcn_embeddings(evolve_embeddings_list)
            # evolve_relations = self.get_tcn_embeddings(evolve_rels_list)
            global_embeddings = self.global_forward(test_graph, use_cuda)
            evolve_embeddings.reverse()
            evolve_rels_list.reverse()
            score_list = self.decoder_ob1.forward(evolve_embeddings, evolve_rels_list, all_triples, mode="test")
            score_gb_list = self.decoder_ob2.forward(global_embeddings, evolve_rels_list, all_triples, mode="test",
                                                     partial_embeding=entity_global_history)

            # score_list = [((1 - self.alpha) * s1 + self.alpha * s2).unsqueeze(2)
            #               for s1, s2 in zip(score_list, score_gb_list)]
            score_list = [s.unsqueeze(2) for s in score_list]
            scores_lc = torch.cat(score_list, dim=2)
            scores_lc = torch.softmax(scores_lc, dim=1)

            score_gb_list = [s.unsqueeze(2) for s in score_gb_list]
            scores_gb = torch.cat(score_gb_list, dim=2)
            scores_gb = torch.softmax(scores_gb, dim=1)
            # scores = torch.max(scores, dim=2)[0]

            # scores = (1 - self.alpha) * torch.sum(scores, dim=-1) +\
            #          self.alpha * torch.sum(scores_gb, dim=-1)

            scores = torch.sum((1 - self.alpha) * scores_lc + self.alpha * scores_gb, dim=-1)

            rscores = torch.zeros_like(scores)

            return scores, rscores

    def get_loss(self, glist, triples, prev_model, use_cuda, entity_global_history=None):
        """
        :param glist:
        :param triplets:
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0, 3]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        evolve_embeddings = []
        # evolve_embeddings_list = []
        evolve_rels_list = []
        for idx in range(len(glist)):
            evolve_embs, evolve_rels = self.forward(glist[idx:], use_cuda)
            # evolve_embeddings_list.append(evolve_embs)
            evolve_rels_list.append(evolve_rels)
            evolve_embeddings.append(evolve_embs[-1])
        # evolve_embeddings = self.get_tcn_embeddings(evolve_embeddings_list)
        # evolve_relations = self.get_tcn_embeddings(evolve_rels_list)
        global_embeddings = self.global_forward(glist, use_cuda)
        # global_embeddings.reverse()
        evolve_embeddings.reverse()
        evolve_rels_list.reverse()
        if self.entity_prediction:
            scores_ob = self.decoder_ob1.forward(evolve_embeddings, evolve_rels_list, all_triples)#.view(-1, self.num_ents)
            scores_gb = self.decoder_ob2.forward(global_embeddings, evolve_rels_list, all_triples, partial_embeding=entity_global_history)

            #.view(-1, self.num_ents)
            # scores = [(1 - self.alpha) * s1 + self.alpha * s2 for s1, s2 in zip(scores_ob, scores_gb)]
            for idx in range(len(scores_ob)):
                # (1 - self.alpha) *
                loss_ent += (1 - self.beta) * self.loss_e(scores_ob[idx], all_triples[:, 2])
            for idx in range(len(scores_gb)):
                loss_ent += self.beta * self.loss_e(scores_gb[idx], all_triples[:, 2])

        return loss_ent

