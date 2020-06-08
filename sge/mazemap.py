import os
import numpy as np
from collections import deque
from sge.utils import MOVE_ACTS, AGENT, BLOCK, WATER, EMPTY, KEY, OBJ_BIAS,\
    TYPE_PICKUP, TYPE_TRANSFORM, \
    WHITE, BLACK, DARK, LIGHT, GREEN, DARK_RED

CHR_WIDTH = 9
TABLE_ICON_SIZE = 40
MARGIN = 10
LEGEND_WIDTH = 250

__PATH__ = os.path.abspath(os.path.dirname(__file__))


class Mazemap(object):
    def __init__(self, game_name, game_config):
        if game_name not in ['playground', 'mining']:
            raise ValueError("Unsupported : {}".format(game_name))

        # load game config (outcome)
        self.gamename = game_name
        self.config = game_config

        self.operation_list = self.config.operation_list
        self.legal_actions = self.config.legal_actions

        self.step_penalty = 0.0
        self.w = game_config.width
        self.h = game_config.height

        # map tensor
        self.obs = np.zeros(
            (self.config.nb_obj_type+3, self.w, self.h), dtype=np.uint8)
        self.wall_mask = np.zeros((self.w, self.h), dtype=np.bool_)
        self.item_map = np.zeros((self.w, self.h), dtype=np.int16)

    def reset(self, subtask_id_list):
        self.subtask_id_list = subtask_id_list
        self.nb_subtask = len(subtask_id_list)
        self.obs.fill(0)
        self.wall_mask.fill(0)
        self.item_map.fill(-1)
        self.empty_list = []

        self._add_blocks()
        self._add_targets()

    def act(self, action):
        oid = -1
        assert action in self.legal_actions, 'Illegal action: '
        if action in {KEY.UP, KEY.DOWN, KEY.LEFT, KEY.RIGHT}:  # move
            new_x = self.agent_x
            new_y = self.agent_y
            if action == KEY.RIGHT:
                new_x += 1
            elif action == KEY.LEFT:
                new_x -= 1
            elif action == KEY.DOWN:
                new_y += 1
            elif action == KEY.UP:
                new_y -= 1
            # wall_collision
            if not (new_x, new_y) in self.walls and not (new_x, new_y) in self.waters:
                self.obs[AGENT, self.agent_x, self.agent_y] = 0
                self.agent_x = new_x
                self.agent_y = new_y
                self.obs[AGENT, new_x, new_y] = 1
        else:  # perform
            iid = self._get_cur_item()
            if iid > -1:
                oid = iid-3
                self._perform(action, oid)  # perform action in the map
        self._process_obj()  # moving objects
        return oid

    def get_obs(self):
        return self.obs

    def _process_obj(self):
        for obj in self.object_list:
            oid = obj['oid']
            obj_param = self.config.object_param_list[oid]
            if 'speed' in obj_param and obj_param['speed'] > 0 and np.random.uniform() < obj_param['speed']:
                # randomly move
                x, y = obj['pos']
                candidates = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                pool = []
                for nx, ny in candidates:
                    if self.item_map[nx, ny] == -1:
                        pool.append((nx, ny))
                if len(pool) > 0:
                    new_pos = tuple(np.random.permutation(pool)[0])
                    # remove and push
                    self._remove_item(obj)
                    self._add_item(oid, new_pos)

    def _remove_item(self, obj):
        oid = obj['oid']
        x, y = obj['pos']
        self.obs[oid+OBJ_BIAS, x, y] = 0
        self.item_map[x, y] = -1
        self.object_list.remove(obj)

    def _add_item(self, oid, pos):
        obj = dict(oid=oid, pos=pos)
        self.obs[oid+OBJ_BIAS, pos[0], pos[1]] = 1
        self.item_map[pos[0], pos[1]] = oid+OBJ_BIAS
        self.object_list.append(obj)

    def _perform(self, action, oid):
        assert(action not in MOVE_ACTS)
        act_type = self.operation_list[action]['oper_type']
        obj = None
        for oind in range(len(self.object_list)):
            o = self.object_list[oind]
            if o['pos'] == (self.agent_x, self.agent_y):
                obj = o
                break
        assert obj is not None

        # pickup
        if act_type == TYPE_PICKUP and self.config.object_param_list[oid]['pickable']:
            self._remove_item(obj)
        # transform
        elif act_type == TYPE_TRANSFORM and self.config.object_param_list[oid]['transformable']:
            self._remove_item(obj)
            outcome_oid = self.config.object_param_list[oid]['outcome']
            self._add_item(outcome_oid, (self.agent_x, self.agent_y))

    def _add_blocks(self):
        # boundary
        self.walls = [(0, y) for y in range(self.h)]  # left wall
        self.walls = self.walls + [(self.w-1, y)
                                   for y in range(self.h)]  # right wall
        self.walls = self.walls + [(x, 0)
                                   for x in range(self.w)]  # bottom wall
        self.walls = self.walls + [(x, self.h-1)
                                   for x in range(self.w)]  # top wall

        for x in range(self.w):
            for y in range(self.h):
                if (x, y) not in self.walls:
                    self.empty_list.append((x, y))
                else:
                    self.item_map[x, y] = 1  # block

        # random block
        if self.config.nb_block[0] < self.config.nb_block[1]:
            nb_block = np.random.randint(
                self.config.nb_block[0], self.config.nb_block[1]+1)
            num_candidate = len(self.empty_list)
            pool = np.random.permutation(self.empty_list)
            pool_idx = 0
            for block_idx in range(nb_block):
                success = False
                while pool_idx < num_candidate:
                    # 1. pop from candidate
                    x, y = pool[pool_idx]
                    pool_idx += 1

                    # 2. check connectivity
                    self.empty_list.remove((x, y))
                    self.item_map[x, y] = BLOCK
                    if self._check_connectivity(self.empty_list): # if okay, add the block
                        self.walls.append((x, y))
                        self.obs[BLOCK, x, y] = 1
                        success = True
                        break
                    else: # if not, revert
                        self.empty_list.append((x, y))
                        self.item_map[x, y] = EMPTY
                if not success:
                    import ipdb; ipdb.set_trace()
                    raise RuntimeError('Cannot generate a map without\
                        inaccessible regions! Decrease the #waters or #blocks')

        for (x, y) in self.walls:
            self.obs[BLOCK, x, y] = 1

        # random water
        self.waters = []
        if self.config.nb_water[0] < self.config.nb_water[1]:
            nb_water = np.random.randint(
                self.config.nb_water[0], self.config.nb_water[1]+1)
            num_candidate = len(self.empty_list)
            pool = np.random.permutation(self.empty_list)
            pool_idx = 0
            
            for water_idx in range(nb_water):
                success = False
                while pool_idx < num_candidate:
                    # 1. pop from candidate
                    x, y = pool[pool_idx]
                    pool_idx += 1

                    # 2. check connectivity
                    self.empty_list.remove((x, y))
                    self.item_map[x, y] = WATER
                    if self._check_connectivity(self.empty_list): # if okay, add the water
                        self.waters.append((x, y))
                        self.obs[WATER, x, y] = 1
                        success = True
                        break
                    else: # if not, revert
                        self.empty_list.append((x, y))
                        self.item_map[x, y] = EMPTY
                if not success:
                    import ipdb; ipdb.set_trace()
                    raise RuntimeError('Cannot generate a map without\
                        inaccessible regions! Decrease the #waters or #blocks')

    def _add_targets(self):
        # reset
        self.object_list = []
        self.omask = np.zeros((self.config.nb_obj_type), dtype=np.int8)

        # create objects
        # 1. create required objects
        pool = np.random.permutation(self.empty_list)
        for tind in range(self.nb_subtask):
            # make sure each subtask is executable
            self._place_object(tind, (pool[tind][0], pool[tind][1]))
        # 2. create additional objects
        index = self.nb_subtask
        for obj_param in self.config.object_param_list:
            if 'max' in obj_param:
                oid = obj_param['oid']
                nb_obj = np.random.randint(0, obj_param['max']+1)
                for i in range(nb_obj):
                    self._add_item(oid, (pool[index][0], pool[index][1]))
                    index += 1

        # create agent
        (self.agent_init_pos_x, self.agent_init_pos_y) = pool[index]
        self.agent_x = self.agent_init_pos_x
        self.agent_y = self.agent_init_pos_y

        self.obs[AGENT, self.agent_x, self.agent_y] = 1

    def _place_object(self, task_ind, pos):
        subid = self.subtask_id_list[task_ind]
        (_, oid) = self.config.subtask_param_list[subid]
        if ('unique' not in self.config.object_param_list[oid]) or \
            (not self.config.object_param_list[oid]['unique']) or \
                (self.omask[oid] == 0):
            self.omask[oid] = 1
            self._add_item(oid, pos)

    def _check_connectivity(self, empty_list):
        nb_empty = len(empty_list)
        mask = np.copy(self.item_map)
        #
        queue = deque([empty_list[0]])
        x, y = empty_list[0]
        mask[x, y] = 1
        count = 0
        while len(queue) > 0:
            [x, y] = queue.popleft()
            count += 1
            candidate = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            for item in candidate:
                if mask[item[0], item[1]] == -1:  # if empty
                    mask[item[0], item[1]] = 1
                    queue.append(item)
        if count > nb_empty:
          print('Bug in the code')
          import ipdb; ipdb.set_trace()
        return count == nb_empty

    def _get_cur_item(self):
        return self.item_map[self.agent_x, self.agent_y]
