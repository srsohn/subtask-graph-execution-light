import os
import numpy as np
import time
from collections import deque
from .utils import TimeProfiler
from sge.utils import MOVE_ACTS, AGENT, BLOCK, WATER, EMPTY, KEY, OBJ_BIAS,\
    TYPE_PICKUP, TYPE_TRANSFORM, MAX_DIST, OID_TO_IID

__PATH__ = os.path.abspath(os.path.dirname(__file__))


class Mazemap(object):
    def __init__(self, game_name, game_config, render=False):
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
        self.num_nodes = (self.w - 2) * (self.h - 2)

        # map tensor
        self._empty_item_map, self._full_adj_mat = self._default_matrices()
        self.obs = np.zeros(
            (self.config.nb_obj_type+3, self.w, self.h), dtype=np.uint8)
        self.item_map = np.zeros((self.w, self.h), dtype=np.int16)

        # Visualization
        self._rendering = render
        self.render_dir = 'render'
        if self._rendering:
          self._load_game_asset()
        self.tpf = TimeProfiler('Map/')

    def reset(self, subtask_id_list, reset_map=True):
        self.subtask_id_list = subtask_id_list
        self.nb_subtask = len(subtask_id_list)
        if reset_map:
            self.obs.fill(0)
            self.item_map.fill(EMPTY)

            empty_list, blocked_list = self._add_blocks()
            self._add_targets(empty_list)
            #
            self._calculate_distance(blocked_list)
            self._save_init_map()
        else:
            self._load_init_map()

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
            item_id = self.item_map[new_x, new_y]
            # If not block or water, agent can move
            if not (item_id == BLOCK or item_id == WATER):
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

    def render(self, task, step, epi=None):
        from imageio import imwrite
        if not self._rendering:
            return
        scale = self.config.rendering_scale
        self.screen = np.ones( (scale*self.w, scale*self.h, 4), dtype=np.uint8 )*255
        self.screen[:, :, 3] = 0
        # Draw items (agent, walls, objs)
        for x in range(self.w):
            for y in range(self.h):
                iid = self.item_map[x, y]
                if iid >=0:
                  item_img = self.image_by_iid[iid]
                  self._draw_cell( (x, y), item_img)
        # Agent in the front
        self._draw_cell( (self.agent_x, self.agent_y), self.image_by_iid[AGENT])
        
        # Grid
        indices = np.concatenate( (np.arange(0, scale*self.w, scale), np.arange(scale-1, scale*self.w, scale) ) )
        self.screen[indices, :, :3] = 0
        self.screen[indices, :, 3] = 255
        indices = np.concatenate( (np.arange(0, scale*self.h, scale), np.arange(scale-1, scale*self.h, scale) ) )
        self.screen[:, indices, :3] = 0
        self.screen[:, indices, 3] = 255
        # File writing
        if self.render_dir is not None:
            im_name = self.render_dir + '/image_task%s'%(task)
            if epi is not None:
              im_name += '_epi=%s'%(epi)
            im_name += '_step=%s.png'%(step)
            imwrite(im_name, self.screen)
    
    def get_obs(self):
        return self.obs

    def move_to_closest_obj(self, oid, max_step):
        cur_pos = (self.agent_y-1) * (self.w - 2) + (self.agent_x-1)
        dist_mat = self.distance[cur_pos]
        # 1. get closest location of object with type 'oid' (within max_step)
        mindist = -1
        obj_x, obj_y = None, None
        for obj in self.object_list:
            if obj['oid'] == oid:
                x, y = obj['pos']
                target_pos = (y-1) * (self.w - 2) + (x-1)
                dist = dist_mat[target_pos]
                assert dist < MAX_DIST, 'Warning! infinite distance!'
                if dist <= max_step and (mindist > dist or mindist < 0):
                    mindist = dist
                    obj_x, obj_y = x, y
        # 2. teleport to it
        if mindist >= 0:
            self.agent_x = obj_x
            self.agent_y = obj_y
        return mindist

    #############
    def _save_init_map(self):
      self._init_object_list = self.object_list.copy()
      self._init_item_map = self.item_map.copy()
      self._init_obs = self.obs.copy()
    
    def _load_init_map(self):
      self.object_list = self._init_object_list.copy()
      np.copyto(self.item_map, self._init_item_map)
      np.copyto(self.obs, self._init_obs)
      self.agent_x = self.agent_init_pos_x
      self.agent_y = self.agent_init_pos_y

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
        ___tpf = self.tpf
        ___tpf.stamp(time.time())
        # empty positions
        init_empty_list = np.argwhere(self._empty_item_map == EMPTY)

        # random block & water
        nb_block = np.random.randint(self.config.nb_block[0], self.config.nb_block[1]+1)
        nb_water = np.random.randint(self.config.nb_water[0], self.config.nb_water[1]+1)
        assert nb_block+nb_water < self.num_nodes //6, 'Too many block and water cells!'
        ___tpf.stamp(time.time(), 'block/Init')

        success = False
        while not success:
            np.copyto(self.item_map, self._empty_item_map)
            ___tpf.stamp(time.time())
            pool = np.random.permutation(init_empty_list)
            # Add all block and water at once
            block_positions = pool[:nb_block].transpose()
            self.item_map[ (block_positions[0], block_positions[1]) ] = BLOCK

            water_positions = pool[nb_block:nb_block+nb_water].transpose()
            self.item_map[ (water_positions[0], water_positions[1]) ] = WATER
            ___tpf.stamp(time.time(), 'block/Etc')

            # test connectivity
            success, empty_list = self._check_connectivity(self.item_map == EMPTY)
            blocked_list = pool[:nb_block+nb_water]
            ___tpf.stamp(time.time(), 'block/Connectivity')

        # Apply changes to self.obs
        ___tpf.stamp(time.time())
        self.obs[BLOCK][self.item_map == BLOCK] = True
        self.obs[WATER][self.item_map == WATER] = True
        ___tpf.stamp(time.time(), 'block/Etc')
        return empty_list, blocked_list

    def _add_targets(self, empty_list):
        # reset
        self.object_list = []
        omask = np.zeros((self.config.nb_obj_type), dtype=np.int8)

        # create objects
        # 1. create required objects
        pool = np.random.permutation(empty_list)
        for tind in range(self.nb_subtask):
            # make sure each subtask is executable
            omask = self._place_object(omask, tind, (pool[tind][0], pool[tind][1]))
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

    def _place_object(self, omask, task_ind, pos):
        subid = self.subtask_id_list[task_ind]
        (_, oid) = self.config.subtask_param_list[subid]
        if ('unique' not in self.config.object_param_list[oid]) or \
            (not self.config.object_param_list[oid]['unique']) or \
                (omask[oid] == 0):
            omask[oid] = 1
            self._add_item(oid, pos)
        return omask

    def _default_matrices(self):
        # Construct adjacency matrix
        width = self.w - 2
        height = self.h - 2
        first, second = [], []
        for x in range(width):
          for y in range(height):
            cur = y * width + x
            if x > 0:
              first.append( cur )
              second.append( cur-1 )
            if x < width-1:
              first.append( cur )
              second.append( cur+1 )
            if y > 0:
              first.append( cur )
              second.append( cur-width )
            if y < height-1:
              first.append( cur )
              second.append( cur+width )
        #
        adj_mat = np.zeros( (width * height, width * height), dtype=np.bool )
        adj_mat[(first, second)] = True
        # 
        empty_item_map = np.zeros((self.w, self.h), dtype=np.int16)
        empty_item_map.fill(EMPTY)
        empty_item_map[0:self.w:self.w-1, :] = BLOCK  # left, right wall
        empty_item_map[:, 0:self.h:self.h-1] = BLOCK  # top, bottom wall

        return empty_item_map, adj_mat

    def _construct_adjacency_matrix(self, blocked_list):
        width = self.w - 2
        height = self.h - 2
        adj_mat = self._full_adj_mat.copy()
        # remove
        first, second = [], []
        for x, y in blocked_list:
          cur = (y-1) * width + x-1
          if x > 0:
            first.append( cur )
            second.append( cur-1 )
          if x < width-1:
            first.append( cur )
            second.append( cur+1 )
          if y > 0:
            first.append( cur )
            second.append( cur-width )
          if y < height-1:
            first.append( cur )
            second.append( cur+width )
        adj_mat[(first, second)] = False
        return adj_mat

    def _calculate_distance(self, blocked_list):
        ___tpf = self.tpf
        ___tpf.stamp(time.time())
        adj_mat = self._construct_adjacency_matrix(blocked_list)
        ___tpf.stamp(time.time(), 'dist/adjacency matrix')

        # Initialize distance matrix
        dist = adj_mat.astype(np.int16)
        dist[~adj_mat] = MAX_DIST
        np.fill_diagonal(dist, 0)
        # Floyd-Warshall
        for k in range(self.num_nodes):
            dist = np.minimum(dist, dist[np.newaxis,k,:] + dist[:,k,np.newaxis])
        self.distance = dist
        ___tpf.stamp(time.time(), 'dist/FW')

    def _check_connectivity(self, empty_map):
        empty_list = np.argwhere(empty_map)
        nb_empty = len(empty_list)
        #
        queue = deque([empty_list[0]])
        x, y = empty_list[0]
        empty_map[x, y] = False
        count = 0
        while len(queue) > 0:
            [x, y] = queue.popleft()
            count += 1
            candidate = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            for item in candidate:
                if empty_map[item[0], item[1]]:  # if empty
                    empty_map[item[0], item[1]] = False
                    queue.append(item)
        if count > nb_empty:
            print('Bug in the code')
            import ipdb; ipdb.set_trace()
        return count == nb_empty, empty_list

    def _get_cur_item(self):
        return self.item_map[self.agent_x, self.agent_y]

    def get_obj_position_by_oid(self, oid):
        for obj in self.object_list:
            if obj['oid'] == oid:
                return obj['pos']
    
    def get_obj_position_by_oid(self, oid):
        for obj in self.object_list:
            if obj['oid'] == oid:
                return obj['pos']

    def _draw_cell(self, pos, obj_img = None):
        scale = self.config.rendering_scale
        if obj_img is not None:
            np.copyto(self.screen[ pos[0]*scale : pos[0]*scale+scale, pos[1]*scale : pos[1]*scale+scale, : ], obj_img  )
          
    def _load_game_asset(self):
        from imageio import imread
        self.image_by_iid = dict()
        #
        item_image_name_by_iid = self.config.item_image_name_by_iid
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        img_folder = os.path.join(ROOT_DIR, 'asset', self.gamename, 'Icon')
        for obj in self.config.object_param_list:
            image = imread( os.path.join(img_folder,obj['imgname']) )
            self.image_by_iid[OID_TO_IID(obj['oid'])] = image
        self.image_by_iid[AGENT] = imread(os.path.join(img_folder, item_image_name_by_iid[AGENT]))
        self.image_by_iid[BLOCK] = imread(os.path.join(img_folder, item_image_name_by_iid[BLOCK]))
        if item_image_name_by_iid[WATER] is not None:
            self.image_by_iid[WATER] = imread(os.path.join(img_folder, item_image_name_by_iid[WATER]))