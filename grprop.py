import os
import torch
from torch.autograd import Variable
import numpy as np
from agents import RandomOptionAgent

class GRProp(RandomOptionAgent):
  def __init__(self, 
               option_space, 
               batch_size, 
               temp=200., 
               beta_a=8., 
               w_a=3., 
               ep_or=0.8, 
               temp_or=2.):
    self.algo = 'grprop'
    self.option_space = option_space
    self.batch_size = batch_size
    #
    self.temp = temp
    self.beta_a = beta_a
    self.w_a = w_a
    self.ep_or = ep_or
    self.temp_or = temp_or

  def update_graph(self, graphs, dones=None):
    if type(graphs) == list:
      self.init_graph(dones, graphs)
    else:
      self.init_batch_graph(dones, graphs)

  def sample_option(self, states, eval_flag=False):
    mask_tensor = torch.stack([torch.from_numpy(s['mask_ind']).float() for s in states])
    comp_tensor = torch.stack([torch.from_numpy(s['comp_ind']).float() for s in states])
    elig_tensor = torch.stack([torch.from_numpy(s['elig_ind']).float() for s in states])

    self.ntasks = len(elig_tensor[0])
    batch_size = self.batch_size

    # 1. prepare input
    x = Variable(comp_tensor.unsqueeze(-1), requires_grad=True)
    r = (self.rew_tensor * mask_tensor).unsqueeze(-1)

    # 2. compute grad
    logits = self._compute_RProp_grad(x, r)

    # if flat, switch to greedy.
    logits += (self.num_layers == 1).view(-1, 1).repeat(1,
               self.ntasks).float()*self.rew_tensor

    # 3. masking
    masked_elig_batch = torch.mul(mask_tensor, elig_tensor).detach()
    if eval_flag:
      logits = logits-torch.min(logits) + 1.0
      tind = torch.argmax(
          torch.mul(logits, masked_elig_batch.float()), dim=1).unsqueeze(-1)
      tid = _transform(tind, self.tind_to_tid).squeeze()
    else:
      masked_logit = logits * masked_elig_batch
      masked_logit = self.temp * \
          (masked_logit - masked_logit.min(1)
           [0].unsqueeze(-1)) * masked_elig_batch
      prob = F.softmax(masked_logit, dim=1).data
      prob_masked = prob * masked_elig_batch
      m = torch.distributions.Categorical(prob_masked)
      tind = m.sample()       # Nbatch x 1
      tid = _transform(tind, self.tind_to_tid).squeeze()

    return tid

  def init_batch_graph(self, dones, graph):
    batch_size = self.batch_size
    self.num_layers = torch.tensor([graph.nlayer]*graph.nbatch)
    self.max_num_layer = graph.nlayer - 1
    max_NA, max_NP = graph.max_NA, graph.max_NP
    self.rew_tensor = graph.rmag
    self.tind_to_tid = graph.ind_to_id

    # Set self.Wa_tensor, Wa_neg, Wo_tensor, Pmask
    self.Wa_neg = graph.ANDmat.lt(0).float()
    self.Wa_tensor = torch.zeros(
        self.max_num_layer, batch_size, max_NA, max_NP)
    self.Wo_tensor = torch.zeros(
        self.max_num_layer, batch_size, max_NP, max_NA)
    self.Pmask = torch.zeros(self.max_num_layer+1, batch_size, max_NP, 1)

    for bind in range(batch_size):
      if dones is None or dones[bind]:
        tind_by_layer = graph.tind_by_layer[bind]
        num_layer = len(tind_by_layer)-1
        ANDmat = graph.ANDmat[bind]
        ORmat = graph.ORmat[bind]
        numa = graph.numA[bind]
        #
        abias = 0
        tind_tensor = torch.LongTensor(tind_by_layer[0])
        mask = torch.zeros(max_NP).scatter_(0, tind_tensor, 1)
        self.Pmask[0, bind].copy_(mask.unsqueeze(-1))
        for lind in range(num_layer):
          # prepare
          if len(numa) < lind+1 or len(graph.numP[bind]) < lind+1:
              import ipdb; ipdb.set_trace()

          # W_a
          na = numa[lind]
          wa = ANDmat[abias:abias+na, :]
          # output is current layer only.
          self.Wa_tensor[lind, bind, abias:abias+na, :].copy_(wa)

          # W_o
          tind = tind_by_layer[lind+1]
          wo = ORmat[:, abias:abias+na]    # numA x numP_cumul

          # re-arrange to the original subtask order
          # input (or) is cumulative. output is current layer only.
          self.Wo_tensor[lind, bind, :, abias:abias+na].copy_(wo)
          abias += na

          tind_tensor = torch.LongTensor(tind)
          mask = torch.zeros(max_NP).scatter_(0, tind_tensor, 1)
          self.Pmask[lind+1, bind].copy_(mask.unsqueeze(-1))
    self.Wa_tensor = self.Wa_tensor.gt(0).float() # only the positive entries

  def init_graph(self, dones, graphs):
    # initialize self.Wa_tensor, Wo_tensor, rew_tensor
    # prepare
    batch_size = len(graphs)
    self.num_layers = torch.tensor([len(g.tind_by_layer) for g in graphs])
    self.max_num_layer = max([len(g.tind_by_layer) for g in graphs]) - 1
    max_NA = max([g.ANDmat.shape[0] for g in graphs])  # max total-#-A
    max_NP = max([len(g.rmag) for g in graphs])  # max total-#-P
    self.rew_tensor = torch.zeros(batch_size, max_NP)
    self.tind_to_tid = torch.zeros(batch_size, max_NP).long()

    for bind, graph in enumerate(graphs):
      if dones is None or dones[bind]:
        self.rew_tensor[bind].copy_(graph.rmag)
        if type(graph.ind_to_id) == dict:
          for k, v in graph.ind_to_id.items():
            self.tind_to_tid[bind][k] = v
        else:
          self.tind_to_tid[bind].copy_(graph.ind_to_id)

    if self.max_num_layer == 0:
      print('Warning!! flat graph!!!')
      self.fast_teacher_mode = 'GR'
    else:
      self.Wa_tensor = torch.zeros(
          self.max_num_layer, batch_size, max_NA, max_NP)
      self.Wa_neg = torch.zeros(batch_size, max_NA, max_NP)
      self.Wo_tensor = torch.zeros(
          self.max_num_layer, batch_size, max_NP, max_NA)
      self.Pmask = torch.zeros(self.max_num_layer+1, batch_size, max_NP, 1)

      for bind, graph in enumerate(graphs):
        if dones is None or dones[bind]:
          tind_by_layer = graph.tind_by_layer
          num_layer = len(tind_by_layer)-1
          #
          if type(graph.rmag) == list:
            graph.rmag = torch.tensor(graph.rmag)
          if num_layer > 0:
            # W_a_neg
            ANDmat = graph.ANDmat
            ORmat = graph.ORmat
            self.Wa_neg[bind, :ANDmat.shape[0], :ANDmat.shape[1]].copy_(
                ANDmat.lt(0).float())  # only the negative entries
            abias, tbias = 0, graph.numP[0]
            tind_tensor = torch.LongTensor(tind_by_layer[0])
            mask = torch.zeros(max_NP).scatter_(0, tind_tensor, 1)
            self.Pmask[0, bind].copy_(mask.unsqueeze(-1))
            for lind in range(num_layer):
              # prepare
              if len(graph.numA) < lind+1 or len(graph.numP) < lind+1:
                import ipdb; ipdb.set_trace()

              # W_a
              na, _ = graph.W_a[lind].shape
              wa = ANDmat[abias:abias+na, :]
              # output is current layer only.
              self.Wa_tensor[lind, bind, abias:abias+na, :].copy_(wa)

              # W_o
              tind = tind_by_layer[lind+1]
              wo = ORmat[:, abias:abias+na]    # numA x numP_cumul
              nt, _ = graph.W_o[lind].shape

              # re-arrange to the original subtask order
              # input (or) is cumulative. output is current layer only.
              self.Wo_tensor[lind, bind, :, abias:abias+na].copy_(wo)
              abias += na

              tind_tensor = torch.LongTensor(tind)
              mask = torch.zeros(max_NP).scatter_(0, tind_tensor, 1)
              self.Pmask[lind+1, bind].copy_(mask.unsqueeze(-1))
              tbias += nt

      self.Wa_tensor = self.Wa_tensor.gt(0).float() # only the positive entries

  def _compute_RProp_grad(self, tp, reward):
    # tp: Nb x NP x 1         : tp input. {0,1}
    # p: precondition progress [0~1]
    # a: output of AND node. I simply added the progress.
    # or: output of OR node.
    # p = softmax(a).*a (soft version of max function)
    # or = max (x, \lambda*p + (1-\lambda)*x). After execution (i.e. x=1), gradient should be blocked. So we use max(x, \cdot)
    # a^ = Wa^{+}*or / Na^{+} + 0.01   -       Wa^{-}*or.
    #     -> in [0~1]. prop to #satisfied precond.       --> If any neg is executed, becomes <0.
    # a = max( a^, 0 ) # If any neg is executed, gradient is blocked
    # Intuitively,
    # p: soft version of max function
    # or: \lambda*p + (1-\lambda)*x
    # a: prop to portion of satisfied precond
    #############

    # 1. forward (45% time)
    or_ = torch.max(tp, self.ep_or + (0.99-self.ep_or)*tp) * self.Pmask[0]
    A_neg = torch.matmul(self.Wa_neg, tp)     #(Nb x NA x NP) * (Nb x NP x 1) = (Nb x NA x 1)
    or_list = [or_]
    for lind in range(self.max_num_layer):
      # Init
      wa = self.Wa_tensor[lind]
      wo = self.Wo_tensor[lind]
      Pmask = self.Pmask[lind+1]

      or_concat = torch.cat( or_list, dim=2 )

      # AND layer
      a_pos = ( torch.matmul(wa, or_concat).sum(-1) / wa.sum(-1).clamp(min=1) ).unsqueeze(-1) # (Nb x NA x 1)
      a_hat = a_pos - self.w_a* A_neg                             #Nb x Na x 1
      and_ = nn.Softplus(self.beta_a)(a_hat)                      #Nb x Na x 1 (element-wise)

      # soft max version2
      num_next_or = wo.shape[1]
      and_rep = and_.transpose(1,2).repeat(1, num_next_or, 1)
      p_next = ( self.masked_softmax(self.temp_or*and_rep, wo) * and_rep ).sum(dim=-1).unsqueeze(-1)

      or_ = torch.max(tp, self.ep_or*p_next + (0.99-self.ep_or)*tp) * Pmask #Nb x Np_sum x 1
      or_list.append(or_)

    # loss (soft reward)
    or_mat = torch.cat(or_list, dim=2)
    soft_reward = torch.matmul(or_mat.transpose(1,2), reward).sum() # (Nb x Nt).*(Nb x Nt) (element-wise multiply)

    # 2. back-prop (45% time)
    soft_reward.backward()

    # 3. mapping from GRProp gradient to task_ind gradient (10% time)
    gradient = tp.grad.squeeze(-1)

    if torch.isnan(gradient).any():
      print('Error! there is a NaN in the gradient')
      import ipdb; ipdb.set_trace()
    return gradient

  def masked_softmax(self, mat, mask, dim=2, epsilon=1e-6):
    nb, nr, nc = mat.shape
    masked_mat = mat * mask
    masked_min = torch.min(masked_mat, dim=dim, keepdim=True)[0].repeat(1, 1, nc)
    masked_nonneg_mat = (masked_mat - masked_min)*mask
    max_mat = torch.max(masked_nonneg_mat, dim=dim, keepdim=True)[0].repeat(1, 1, nc)
    exps = torch.exp(masked_nonneg_mat-max_mat)
    masked_exps = exps * mask
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    prob =  masked_exps/masked_sums
    return prob
