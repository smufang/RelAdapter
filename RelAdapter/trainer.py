from models import *
from tensorboardX import SummaryWriter
import os
import sys
import torch
import shutil
import logging
import numpy as np

class Trainer:
    def __init__(self, data_loaders, dataset, parameter):
        self.parameter = parameter
        # data loader
        self.train_data_loader = data_loaders[0]
        self.dev_data_loader = data_loaders[1]
        self.test_data_loader = data_loaders[2]
        # parameters
        self.few = parameter['few']
        self.num_query = parameter['num_query']
        self.batch_size = parameter['batch_size']
        self.learning_rate = parameter['learning_rate']
        self.adaptor_learning_rate = parameter['adaptor_learning_rate']
        self.epoch = parameter['epoch']
        self.print_epoch = parameter['print_epoch']
        self.eval_epoch = parameter['eval_epoch']
        self.checkpoint_epoch = parameter['checkpoint_epoch']
        self.early_stopping_patience = parameter['early_stopping_patience']
        # device
        self.device = parameter['device']
        self.metatrain_adaptor = parameter['metatrain_adaptor']
        self.mu = parameter['mu']

        self.metaR = MetaR(dataset, parameter).cuda()
        self.adaptor = WayGAN2(parameter).getVariables2().cuda()
        self.metaR.to(self.device)
        self.adaptor.to(self.device)


        # optimizer
        self.optimizer = torch.optim.Adam(self.metaR.parameters(), self.learning_rate)
        self.optimizer_adaptor = torch.optim.Adam(self.adaptor.parameters(), self.adaptor_learning_rate)


        # tensorboard log writer
        if parameter['step'] == 'train':
            self.writer = SummaryWriter(os.path.join(parameter['log_dir'], parameter['prefix']))
        # dir
        self.state_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'])
        if not os.path.isdir(self.state_dir):
            os.makedirs(self.state_dir)
        self.ckpt_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'], 'checkpoint')
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.state_dict_file = ''

        # logging
        logging_dir = os.path.join(self.parameter['log_dir'], self.parameter['prefix'], 'res.log')
        logging.basicConfig(filename=logging_dir, level=logging.INFO, format="%(asctime)s - %(message)s")

        # load state_dict and params
        if parameter['step'] in ['test', 'dev']:
            self.reload()
            self.reload_adaptor()

    def reload(self):
        if self.parameter['eval_ckpt'] is not None:
            state_dict_file = os.path.join(self.ckpt_dir, 'state_dict_' + self.parameter['eval_ckpt'] + '.ckpt')
        else:
            state_dict_file = os.path.join(self.state_dir, 'state_dict')
        self.state_dict_file = state_dict_file
        logging.info('Reload state_dict from {}'.format(state_dict_file))
        print('reload state_dict from {}'.format(state_dict_file))
        state = torch.load(state_dict_file, map_location=self.device)
        if os.path.isfile(state_dict_file):
            self.metaR.load_state_dict(state)

            #Add in Context info#
            print('loading rel context ... ...')
            entcontext = torch.from_numpy(np.load(data_dir['ent_context'])).to(self.device)
            Context_embed = (1-self.mu)*self.metaR.embedding.embedding.weight + self.mu*entcontext
            self.metaR.embedding.embedding.weight.data.copy_(Context_embed)

        else:
            raise RuntimeError('No state dict in {}!'.format(state_dict_file))

    def reload_adaptor(self):

        state_dict_file_adaptor = os.path.join(self.state_dir, 'state_dict_adaptor')
        self.state_dict_file_adaptor = state_dict_file_adaptor
        logging.info('Reload state_dict_adaptor from {}'.format(state_dict_file_adaptor))
        print('reload state_dict_adaptor from {}'.format(state_dict_file_adaptor))
        state_adaptor = torch.load(state_dict_file_adaptor, map_location=self.device)
        if os.path.isfile(state_dict_file_adaptor) :
            self.adaptor.load_state_dict(state_adaptor)
        else:
            raise RuntimeError('No state dict in {}!'.format(state_dict_file_adaptor))

    def reload_relation_adaptor(self):

        state_dict_file_adaptor = os.path.join(self.state_dir, 'state_relation_adaptor')
        self.state_dict_file_adaptor = state_dict_file_adaptor
        logging.info('Reload state_relation_adaptor from {}'.format(state_dict_file_adaptor))
        print('reload state_relation_adaptor from {}'.format(state_dict_file_adaptor))
        state_adaptor = torch.load(state_dict_file_adaptor, map_location=self.device)
        if os.path.isfile(state_dict_file_adaptor) :
            self.adaptor.load_state_dict(state_adaptor)
        else:
            raise RuntimeError('No state dict in {}!'.format(state_dict_file_adaptor))

    def save_checkpoint(self, epoch):
        torch.save(self.metaR.state_dict(), os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt'))

    def save_checkpoint_adaptor(self, epoch):
        torch.save(self.adaptor.state_dict(), os.path.join(self.ckpt_dir, 'state_dict_adaptor_' + str(epoch) + '.ckpt'))

    def save_relation_adaptor(self, epoch):
        torch.save(self.adaptor.state_dict(), os.path.join(self.ckpt_dir, 'state_relation_adaptor_' + str(epoch) + '.ckpt'))

    def del_checkpoint(self, epoch):
        path = os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt')
        if os.path.exists(path):
            os.remove(path)
        else:
            raise RuntimeError('No such checkpoint to delete: {}'.format(path))

    def save_best_state_dict(self, best_epoch):
        shutil.copy(os.path.join(self.ckpt_dir, 'state_dict_' + str(best_epoch) + '.ckpt'),
                    os.path.join(self.state_dir, 'state_dict'))

    def save_best_adaptor_state_dict(self, best_epoch):
        shutil.copy(os.path.join(self.ckpt_dir, 'state_dict_adaptor_' + str(best_epoch) + '.ckpt'),
                    os.path.join(self.state_dir, 'state_dict_adaptor'))

    def save_best_adaptor_relation(self, best_epoch):
        shutil.copy(os.path.join(self.ckpt_dir, 'state_relation_adaptor_' + str(best_epoch) + '.ckpt'),
                    os.path.join(self.state_dir, 'state_relation_adaptor'))

    def write_training_log(self, data, epoch):
        self.writer.add_scalar('Training_Loss', data['Loss'], epoch)

    def write_validating_log(self, data, epoch):
        self.writer.add_scalar('Validating_MRR', data['MRR'], epoch)
        self.writer.add_scalar('Validating_Hits_10', data['Hits@10'], epoch)
        self.writer.add_scalar('Validating_Hits_5', data['Hits@5'], epoch)
        self.writer.add_scalar('Validating_Hits_1', data['Hits@1'], epoch)

    def logging_training_data(self, data, epoch):
        logging.info("Epoch: {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                      epoch, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def logging_eval_data(self, data, state_path, istest=False):
        setname = 'dev set'
        if istest:
            setname = 'test set'
        logging.info("Eval {} on {}".format(state_path, setname))
        logging.info("MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                      data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))
            
    def rank_predict(self, data, x, ranks):
        # query_idx is the idx of positive score
        query_idx = x.shape[0] - 1
        # sort all scores with descending, because more plausible triple has higher score
        _, idx = torch.sort(x, descending=True)
        rank = list(idx.cpu().numpy()).index(query_idx) + 1
        ranks.append(rank)
        # update data
        if rank <= 10:
            data['Hits@10'] += 1
        if rank <= 5:
            data['Hits@5'] += 1
        if rank == 1:
            data['Hits@1'] += 1
        data['MRR'] += 1.0 / rank

    def do_one_step(self, task, adaptor,iseval=False, curr_rel=''):
        loss, p_score, n_score = 0, 0, 0
        if not iseval:
            self.optimizer.zero_grad()
            p_score, n_score = self.metaR(task,adaptor, iseval, curr_rel)
            y = torch.Tensor([1]).to(self.device)
            loss = self.metaR.loss_func(p_score, n_score, y)
            loss.backward()
            self.optimizer.step()
        elif curr_rel != '':
            p_score, n_score = self.metaR(task, adaptor,iseval, curr_rel)
            y = torch.Tensor([1]).to(self.device)
            loss = self.metaR.loss_func(p_score, n_score, y)
        return loss, p_score, n_score

    def task_training(self, best_value,epoch, best_epoch,task, adaptor,iseval=False, curr_rel=''):
        loss, p_score, n_score = 0, 0, 0
        self.optimizer_adaptor.zero_grad()
        p_score, n_score = self.metaR.task_adaptor(task, adaptor, iseval, curr_rel)
        y = torch.Tensor([1]).to(self.device)
        loss = self.metaR.loss_func(p_score, n_score, y)
        # print(loss)
        loss.backward()
        # print(loss)
        self.optimizer_adaptor.step()

        return

    def do_one_step_eval_by_relation(self, task, adaptor, iseval=False, curr_rel=''):
        loss, p_score, n_score = 0, 0, 0

        p_score, n_score = self.metaR.forward_adaptor(task, adaptor, iseval, curr_rel)
        y = torch.Tensor([1]).to(self.device)
        loss = self.metaR.loss_func(p_score, n_score, y)
        return loss, p_score, n_score


    def train(self):
        # initialization
        best_epoch = 0
        best_value = 0
        bad_counts = 0

        # training by epoch
        for e in range(self.epoch):
            # sample one batch from data_loader
            train_task, curr_rel = self.train_data_loader.next_batch()
            loss, _, _ = self.do_one_step(train_task,self.adaptor, iseval=False, curr_rel=curr_rel)
            # print the loss on specific epoch
            if e % self.print_epoch == 0:
                loss_num = loss.item()
                self.write_training_log({'Loss': loss_num}, e)
                print("Epoch: {}\tLoss: {:.4f}".format(e, loss_num))
            # save checkpoint on specific epoch
            if e % self.checkpoint_epoch == 0 and e != 0:
                print('Epoch  {} has finished, saving...'.format(e))
                self.save_checkpoint(e)
            # do evaluation on specific epoch
            if e % self.eval_epoch == 0 and e != 0:
                print('Epoch  {} has finished, validating...'.format(e))

                valid_data = self.eval(istest=False, epoch=e)
                self.write_validating_log(valid_data, e)

                metric = self.parameter['metric']
                # early stopping checking
                if valid_data[metric] > best_value:
                    best_value = valid_data[metric]
                    best_epoch = e
                    print('\tBest model | {0} of valid set is {1:.3f}'.format(metric, best_value))
                    bad_counts = 0
                    # save current best
                    self.save_checkpoint(best_epoch)
                    self.save_checkpoint_adaptor(best_epoch)
                else:
                    print('\tBest {0} of valid set is {1:.3f} at {2} | bad count is {3}'.format(
                        metric, best_value, best_epoch, bad_counts))
                    bad_counts += 1

                if bad_counts >= self.early_stopping_patience:
                    print('\tEarly stopping at epoch %d' % e)
                    break

        print('Training has finished')
        print('\tBest epoch is {0} | {1} of valid set is {2:.3f}'.format(best_epoch, metric, best_value))
        self.save_best_state_dict(best_epoch)
        self.save_best_adaptor_state_dict(best_epoch)
        print('Finish')

    def eval(self, istest=False, epoch=None):
        self.metaR.eval()
        # clear sharing rel_q
        self.metaR.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        # initial return data of validation
        data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
        ranks = []
        valid_MRR10 = 0
        t = 0
        temp = dict()
        while True:
            # sample all the eval tasks
            eval_task, curr_rel = data_loader.next_one_on_eval()
            # at the end of sample tasks, a symbol 'EOT' will return
            if eval_task == 'EOT':
                break
            t += 1

            _, p_score, n_score = self.do_one_step(eval_task, self.adaptor, iseval=True, curr_rel=curr_rel)

            x = torch.cat([n_score, p_score], 1).squeeze()

            self.rank_predict(data, x, ranks)

            # print current temp data dynamically
            for k in data.keys():
                temp[k] = data[k] / t
            sys.stdout.write("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
            sys.stdout.flush()


        #Calculate MRR@10
        filt_ranks_tot = [x for x in ranks if x <= 10]
        count_below10 = len(filt_ranks_tot)
        for i in filt_ranks_tot:
            valid_MRR10 += 1.0 / i
        MRR_10_valid = valid_MRR10/count_below10

        # print overall evaluation result and return it
        for k in data.keys():
            data[k] = round(data[k] / t, 3)

        if self.parameter['step'] == 'train':
            self.logging_training_data(data, epoch)
        else:
            self.logging_eval_data(data, self.state_dict_file, istest)

        print("{}\tMRR@10: {:.3f}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            [count_below10,t], MRR_10_valid, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))
        return data

    def eval_by_relation(self, istest=False, epoch=None):
        self.metaR.eval()
        self.metaR.rel_q_sharing = dict()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        all_data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
        all_t = 0
        all_ranks = []
        tot_MRR10_all = 0

        for rel in data_loader.all_rels:
            print("rel: {}, num_cands: {}, num_tasks:{}".format(
                rel, len(data_loader.rel2candidates[rel]), len(data_loader.tasks[rel][self.few:])))

            if self.metatrain_adaptor == 'False':
                nn.init.xavier_normal_(self.adaptor.MLP1.fc.weight)
                nn.init.xavier_normal_(self.adaptor.output.fc.weight)
            elif self.metatrain_adaptor == 'True':
                #Initialize Adaptor , Model parameters initlialized earlier
                self.reload_adaptor()

            best_epoch = 0
            best_value = 0


            eval_task_support, curr_rel = data_loader.next_one_on_eval_by_relation_support(rel)
            for epoch in range(50):
                # data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
                # ranks = []


                # Retrieve validation triplet after support
                self.task_training(best_value, epoch, best_epoch, eval_task_support, self.adaptor,
                                                            iseval=True, curr_rel='')

            #Load best Adaptor
            # self.reload_relation_adaptor()
            #Wash metrics for query evaluation
            data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
            temp = dict()
            t = 0
            ranks = []
            tot_MRR10 = 0
            while True:
                #Retrieve query triplet after support + validation
                eval_task, curr_rel = data_loader.next_one_on_eval_by_relation(rel)
                if eval_task == 'EOT':
                    break
                t += 1

                _, p_score, n_score = self.do_one_step_eval_by_relation(eval_task, self.adaptor, iseval=True, curr_rel='')
                ####

                # _, p_score, n_score = self.do_one_step(eval_task, iseval=True, curr_rel=rel)
                x = torch.cat([n_score, p_score], 1).squeeze()

                self.rank_predict(data, x, ranks)

                for k in data.keys():
                    temp[k] = data[k] / t

            #Calculate MRR@10
            filt_ranks = [x for x in ranks if x <= 10]
            count_below10 = len(filt_ranks)
            for i in filt_ranks:
                tot_MRR10 += 1.0 / i

            if count_below10 >0:
                MRR_10 = tot_MRR10/count_below10
            else:
                MRR_10 = 0.0

            print("{}\tMRR@10: {:.3f}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                   [count_below10,t],MRR_10, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))

            for k in data.keys():
                all_data[k] += data[k]
            all_t += t
            all_ranks.extend(ranks)


        #Calculate MRR@10
        filt_ranks_tot = [x for x in all_ranks if x <= 10]
        count_below10 = len(filt_ranks_tot)
        for i in filt_ranks_tot:
            tot_MRR10_all += 1.0 / i
        MRR_10 = tot_MRR10_all/count_below10

        print('Overall')
        for k in all_data.keys():
            all_data[k] = round(all_data[k] / all_t, 3)
        print("{}\tMRR@10: {:.3f}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            [count_below10,all_t], MRR_10, all_data['MRR'], all_data['Hits@10'], all_data['Hits@5'], all_data['Hits@1']))

        return all_data
