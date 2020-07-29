import os
import numpy as np


__PATH__ = os.path.abspath(os.path.dirname(__file__))


class SubtaskGraph(object):
    def __init__(self, folder, filename, max_task=0):
        # 1. init/load graph
        self.filename = filename

        # subtask_list / edges (ANDmat&ORmat, allow) / subtask reward
        self._load_graph(folder)
        self.max_task = max_task
        self.graph_index = -1

    def _load_graph(self, folder):
        fname = os.path.join(__PATH__, folder, self.filename+'.npy')
        self.graph_list = np.load(fname, allow_pickle=True)
        self.num_graph = len(self.graph_list)

    def set_graph_index(self, graph_index):
        self.graph_index = graph_index
        graph = self.graph_list[graph_index]
        self.num_level = len(graph['W_a'])
        self.ANDmat = graph['ANDmat'].astype(np.float)
        self.ORmat = graph['ORmat'].astype(np.float)
        self.W_a = graph['W_a']
        self.W_o = graph['W_o']
        self.nb_subtask = self.ORmat.shape[0]

        self.numP = [self.W_a[0].shape[1]]
        self.numA = []
        self.num_or = self.numP[0]
        self.num_and = 0
        for lv in range(self.num_level):
            self.numP.append(self.W_o[lv].shape[0])
            self.numA.append(self.W_a[lv].shape[0])
            self.num_or = self.num_or + self.numP[lv + 1]
            self.num_and = self.num_and + self.numA[lv]

        self.b_AND = np.not_equal(self.ANDmat, 0).sum(1).astype(np.float)
        self.b_OR = np.ones(self.nb_subtask)
        self.b_OR[:self.numP[0]] = 0

        self.rew_mag = graph['rmag']
        self.subtask_id_list = (graph['trind']).tolist()

        self.ind_to_id = dict()
        self.id_to_ind = dict()
        for ind in range(self.nb_subtask):
            id = self.subtask_id_list[ind]
            self.ind_to_id[ind] = id
            self.id_to_ind[id] = ind

    def get_elig(self, completion):
        ANDmat = self.ANDmat
        b_AND = self.b_AND
        ORmat = self.ORmat
        b_OR = self.b_OR

        tp = completion.astype(np.float)*2-1  # \in {-1,1}
        # sign(A x tp + b) (+1 or 0)
        ANDout = np.not_equal(
            np.sign((ANDmat.dot(tp)-b_AND)), -1).astype(np.float)
        elig = np.not_equal(np.sign((ORmat.dot(ANDout)-b_OR)), -1)
        return elig
