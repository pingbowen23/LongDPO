"""
The Node and MCTS class for AlphaZero.
"""

#
import copy
import json
import math

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Type
from distributed.utils import print_rank_0, print_with_rank
from envs.base_env import CoTEnv
import pdb
from tqdm import tqdm
import heapq
from loguru import logger
import random

class Node(object):
    """
    Overview:
        The node base class for tree_search.
    """

    def __init__(
        self, parent: "Node" = None, prior_p: float = 1.0, initial_value: float = 0.0
    ) -> None:
        self._parent = parent
        self._children = {}
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = prior_p
        self.prior_p_ori = prior_p

        self._initial_value = initial_value
        self._terminated = False

    def __lt__(self, other):
        return self._initial_value < other._initial_value

    @property
    def terminated(self):
        return self._terminated

    def set_as_terminate_node(self):
        self._terminated = True

    @property
    def value(self) -> float:
        """
        Overview:
            The value of the current node.
        Returns:
            - output (:obj:`Int`): Current value, used to compute ucb score.
        """
        if self._visit_count == 0:
            # if not visited, return the initial value
            return self._initial_value
        return self._value_sum / self._visit_count

    def update(self, value: float) -> None:
        """
        Overview:
            Updata the current node information, such as visit_count and value_sum.
        Arguments:
            - value (:obj:`Int`): The value of the node.
        """
        self._visit_count += 1
        self._value_sum += value

    def update_recursive(self, leaf_value: float, mcts_mode: str) -> None:
        """
        Overview:
            Update node information recursively.
        Arguments:
            - leaf_value (:obj:`Int`): The value of the node.
        """
        if mcts_mode == "self_play_mode":
            self.update(leaf_value)
            if self.is_root():
                return
            self._parent.update_recursive(-leaf_value, mcts_mode)
        if mcts_mode == "play_with_bot_mode":
            self.update(leaf_value)
            if self.is_root():
                return
            self._parent.update_recursive(leaf_value, mcts_mode)

    def is_leaf(self) -> Dict:
        """
        Overview:
            Check if the current node is a leaf node or not.
        Returns:
            - output (:obj:`Dict`): Dict type children node.
        """
        return self._children == {}

    def is_root(self) -> bool:
        """
        Overview:
            Check if the current node is a root node or not.
        Returns:
            - output (:obj:`Bool`): Whether it is the parent node.
        """
        return self._parent is None

    @property
    def parent(self) -> None:
        return self._parent

    @property
    def children(self) -> None:
        return self._children

    @property
    def visit_count(self) -> None:
        return self._visit_count

    def get_info(self):
        # return [
        #     "visit_cnt: {}, value: {:.6f}, prior: {:.6f}".format(
        #         self.visit_count, self.value, self.prior_p)
        # ]
        return {
            "visit_cnt": self.visit_count,
            "value": self.value,
            "prior_p": float(self.prior_p_ori),
            "initial_value": self._initial_value,
            "terminated": self.terminated,
        }

    def clear(self):
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = self.prior_p_ori

    def to_json(self):
        childrens = {}
        for name, child_node in self.children.items():
            childrens[name] = child_node.to_json()

        rets = {"children": childrens, "info": self.get_info()}
        return rets
    
    def __str__(self) -> str:
        if self.is_root():
            return "root"
        else:
            return "child: value: {:.3f}, prior: {:.3f}".format(
                self.last_action, self.value, self.prior_p
            )


class LanguageNode(Node):
    text_state: Optional[str] = None
    last_action: Optional[str] = None
    num_generated_token: Optional[int] = None

    def __init__(
        self,
        parent: Node = None,
        prior_p: float = 1.0,
        prm_value: Optional[float] = None,
        text_state: Optional[str] = None,
        last_action: Optional[str] = None,
        initial_value: float = 0.0,
        num_generated_token: Optional[int] = None,
    ) -> None:
        super().__init__(parent, prior_p, initial_value)
        self.text_state = text_state
        self.last_action = last_action
        self.prm_value = prm_value

        self.num_generated_token = num_generated_token
        self.has_collected_token_num = False
        self.depth = 0

    def get_path(self):
        ans = []
        node = self
        while not node.is_root():
            ans.append(node.last_action)
            node = node.parent
        return "\n".join(reversed(ans))

    def get_info(self):
        info_dict = super().get_info()
        if not self.is_root():
            info_dict["last_action"] = self.last_action
            info_dict["prm_value"] = self.prm_value
        else:
            info_dict["text_state"] = self.text_state
        return info_dict
    
    def __str__(self):
        if self.is_root():
            return "root: {}".format(self.text_state)
        else: 
            return "action: {}, value: {:.3f}, prior: {:.3f}".format(
                self.last_action, self.value, self.prior_p
            )


def get_root(node: Node):
    while not node.is_root():
        node = node.parent
    return node


def save_step(prompt,chosen,actions,ucb_scores,depth,answer_role,group_id,save_step_dir,save_refine_dir,prompt_role="added_user"):
    # 保存step-wise数据
    ucb_score_threshold = 6.0
    max_reward = max(ucb_scores)
    
    cadidates = [item["action"] for item in actions if item["action"] != chosen]
    
    if len(cadidates) == 0: #TODO cadidates is empty, no need to save, 整个step wise应该都放弃
        words = chosen.split()
        for i in range(len(words) - 1):
            # 以一定概率交换相邻单词
            swap_prob = 0.3
            if random.random() < swap_prob:
                words[i], words[i + 1] = words[i + 1], words[i]
        rejected = ' '.join(words)
    else:
        rejected = random.choice(cadidates)
    item = {"instruction":prompt,"chosen":chosen,"rejected":rejected,"group_id":group_id,"depth":depth,"answer_role":answer_role,"prompt_role":prompt_role}

    # 指定文件名
    filename = save_step_dir
    # 将item追加写入JSONL文件
    with open(filename, 'a') as file:
        file.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    if save_refine_dir is not None and max_reward < ucb_score_threshold:
        refine_file_name = save_refine_dir
        addtional_candidates = [item["action"] for item in actions if item["action"]!= chosen and item["action"]!= rejected]
        with open(filename, 'a') as file:
            item = {"instruction":prompt,"chosen":chosen,"rejected":rejected,"group_id":group_id,"depth":depth,"addtional_candidates":addtional_candidates}
            file.write(json.dumps(item, ensure_ascii=False) + '\n')    

class SearchTree:
    """
    Overview:
        MCTS search process.
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg

        self._num_simulations = self._cfg.get("num_simulations", 20)

        # UCB formula
        self._pb_c_base = self._cfg.get("pb_c_base", 19652)  # 19652
        self._pb_c_init = self._cfg.get("pb_c_init", 1.25)  # 1.25

        # Root prior exploration noise.
        self._root_dirichlet_alpha = self._cfg.get(
            "root_dirichlet_alpha", 0.3
        )  # 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
        self._root_noise_weight = self._cfg.get("root_noise_weight", 0.25)  # 0.25

        self.root = None

        self.answers = set()
        self.wrong_answers = set()
        self.visited_paths = None

        self.no_terminal_reward = self._cfg.get("no_terminal_reward", True)
        self.mask_non_terminal_node_value = self._cfg.get(
            "mask_non_terminal_node_value", False
        )

        self._init_critic_value = self._cfg.get("init_critic_value", True)

        self._completion_tokens = 0

    @property
    def num_generated_token(self):
        return self._completion_tokens

    def clear_node(self, node):
        assert node is not None
        node.clear()
        for child in node.children.values():
            self.clear_node(child)

    def get_next_action(
        self,
        simulate_env: Type[CoTEnv],
        reward_fn: Optional[Callable] = None,
        temperature: int = 1.0,
        sample: bool = True,
        return_tree=False,
    ) -> Tuple[int, List[float]]:
        """
        Overview:
            calculate the move probabilities based on visit counts at the root node.
        Arguments:
            - simulate_env (:obj:`Class BaseGameEnv`): The class of simulate env.
            - reward_fn (:obj:`Function`): The Callable to compute the state value.
            - temperature (:obj:`Int`): Temperature is a parameter that controls the "softness" of the probability distribution.
            - sample (:obj:`Bool`): The value of the node.
        Returns:
            - action (:obj:`Bool`): Select the action with the most visits as the final action.
            - action_probs (:obj:`List`): The output probability of each action.
        """
        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state())
            self._expand_leaf_node(root, simulate_env, reward_fn)
            self.root = root
        else:
            root = self.root

        if root.is_leaf():
            # if root is leaf node, expand it
            # We have updated the environment legal action when we test the node is leaf node
            # So the expansion won't have bugs
            self._expand_leaf_node(root, simulate_env, reward_fn)

        if sample:
            self._add_exploration_noise(root)

        for n in range(self._num_simulations):
            simulate_env_copy = simulate_env.copy()
            simulate_env_copy.battle_mode = simulate_env_copy.mcts_mode
            self._simulate(root, simulate_env_copy, reward_fn)

        # for debugging
        # print('after simulation')
        # print('value= {}'.format([(k, v.value) for k,v in root.children.items()]))
        # print('visit_count= {}'.format([(k, v.visit_count) for k,v in root.children.items()]))

        action_visits = []
        for action_dict in simulate_env.legal_actions:
            action = action_dict["action"]
            if action in root.children:
                action_visits.append((action, root.children[action].visit_count))
            else:
                action_visits.append((action, 0))

        actions, visits = zip(*action_visits)
        action_probs = nn.functional.softmax(
            1.0
            / temperature
            * np.log(torch.as_tensor(visits, dtype=torch.float32) + 1e-10),
            dim=0,
        ).numpy()
        if sample:
            action = np.random.choice(actions, p=action_probs)
            self.reset_prior(root)
        else:
            action = actions[np.argmax(action_probs)]

        self.root = root
        if return_tree:
            return action, action_probs, root
        return action, action_probs

    def vanila_mcts(
        self,
        simulate_env: Type[CoTEnv],
        num_path: int,
        reward_model_fn: Optional[Callable] = None,
        select_by_prior: bool = False,
        max_depth: Optional[int] = None,
        group_id: Optional[int] = None,
        save_step_dir: Optional[str] = None,
        save_refine_dir: Optional[str] = None,
        target_length: Optional[int] = None,
    ) -> List[Dict]:
        api_call_completion_tokens = 0
        _, info = simulate_env.reset(update_legal_action=False)
        # api_call_completion_tokens += info["api_completion_token"]
        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state())
            self._expand_leaf_node(root, simulate_env, reward_model_fn,target_length=target_length)
            self.root = root
            self.root.depth += 1

        traj_list = []
        # TODO(ziyu): split with 1. select 2. expand 3. rollout 4. backprop
        #  for here is split the for loop with select and rollout
        #  so that arbitrary rollout function can be used here.
        for i_path in range(num_path):
            node = self.root
            env_copy = simulate_env.copy()
            done = False
            while not done:
                if node.visit_count > 0: # node.text_state
                    # if node is visited, select the child with the highest UCB score
                    action, node , ucb_scores, finish_reason = self._select_child(node, env_copy) # env_copy.legal_actions
                else:
                    # choose rollout policy
                    if select_by_prior:
                        # select with prior probability
                        action, node = self._select_by_prior(node, env_copy)
                    else:
                        # select with highest value, since visit_count = 0 in self.ucb 
                        #  will select node with highest value
                        action, node, ucb_scores, finish_reason = self._select_child(node, env_copy)
                
                if finish_reason == "stop":
                    break
                
                # sync terminated flag here
                # XXX(ziyu): find a more clean way                
                env_copy._next_state_terminated = {}
                assert node.last_action == action
                env_copy._next_state_terminated[action] = node.terminated

                if self.root.depth == max_depth or done:
                    answer_role = "assistant"
                else:
                    answer_role = "added_assistant"
                
                _, _, terminated, truncated, info, legal_actions = env_copy.step(
                    action,ucb_scores,find_fact=True, update_legal_action=node.is_leaf()
                )
                done = terminated or truncated
                
                save_step(node.text_state, action, legal_actions,ucb_scores,self.root.depth,answer_role,group_id,save_step_dir=save_step_dir,save_refine_dir=save_refine_dir)                    
                
                if not done and node.is_leaf():
                    self.root.depth += 1
                    self._expand_leaf_node(node, env_copy, reward_model_fn,self.root.depth,target_length)
                    
                # record api_tokens, if not expand, info["api_completion_token"] is 0
                api_call_completion_tokens += info["api_completion_token"]
            else:
                if node.visit_count > 0:
                    leaf_value = node.value
                else:
                    if self._init_critic_value:
                        leaf_value = node._initial_value
                    else:
                        leaf_value = reward_model_fn(env_copy.get_state()).item()
            # node.update_recursive(leaf_value, env_copy.mcts_mode) #TODO, back-propagation is not implemented here.
            '''
            traj_data = {
                "path_idx": i_path,
                "text": env_copy.answer,
                "value": leaf_value,
                "api_completion_tokens": api_call_completion_tokens,
                "tree_completion_tokens": self._completion_tokens,
            }
            traj_list.append(traj_data)
            
            # reset api_call_completion_tokens
            api_call_completion_tokens = 0
            '''
            traj_list = []
        return traj_list


    def _simulate(
        self,
        node: Node,
        simulate_env: Type[CoTEnv],
        reward_fn: Optional[Callable] = None,
    ) -> None:
        """
        Overview:
            Run a single playout from the root to the leaf, getting a value at the leaf and propagating it back through its parents.
            State is modified in-place, so a deepcopy must be provided.
        Arguments:
            - node (:obj:`Class Node`): Current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): The class of simulate env.
            - reward_fn (:obj:`Function`): The Callable to compute the action probs and state value.
        """
        # XXX: fix the bug temporally, better implementation is required.
        winner = None
        done = False
        while not node.is_leaf():
            action, node = self._select_child(node, simulate_env)
            _, _, terminated, truncated, info = simulate_env.step(
                action, update_legal_action=(node.is_leaf() and node.visit_count == 1)
            )
            done = terminated or truncated

            # In original AlphaZero, the leaf node will be expanded once it is reached
            # In our setting, computing legal action is computational inefficient
            # Thus when we reach a leaf node, we will not directly expand it
            # Until the next time, when this node's children are required to be selected
            # In this case, node is leaf node and the visit count number of node is 1
            # Then we expand it

            if not done and node.is_leaf() and node.visit_count == 1:
                # Once we expand the node, the node will not be leaf node any more
                # And the while won't break
                self._expand_leaf_node(node, simulate_env, reward_fn)

            winner = info["winner"]
        """
        in ``self_play_mode``, the leaf_value is calculated from the perspective of player ``simulate_env.current_player``.
        in ``play_with_bot_mode``, the leaf_value is calculated from the perspective of player 1.
        """
        if not done:
            # leaf_value = self._expand_leaf_node(node, simulate_env,
            #                                     reward_fn)

            if not done and self.mask_non_terminal_node_value:
                leaf_value = 0.0
            else:
                if not self._init_critic_value:
                    leaf_value = reward_fn(simulate_env.get_state()).item()
                else:
                    leaf_value = node._initial_value
        else:
            if not self.no_terminal_reward:
                if winner is not None:
                    if winner == 1:
                        self.answers.add(simulate_env.answer)
                    else:
                        self.wrong_answers.add(simulate_env.answer)

                # if simulate_env.mcts_mode == 'self_play_mode':
                #     if winner == -1:
                #         leaf_value = 0
                #     else:
                #         leaf_value = 1 if simulate_env.current_player == winner else -1

                if simulate_env.mcts_mode == "play_with_bot_mode":
                    # in ``play_with_bot_mode``, the leaf_value should be transformed to the perspective of player 1.
                    if "reward" in info.keys():
                        leaf_value = info["reward"]
                    else:
                        if winner == -1:
                            leaf_value = 0
                        elif winner == 1:
                            leaf_value = 1
                        elif winner == 2:
                            leaf_value = -1
            else:
                if node.visit_count > 0:
                    # because leaf value has been calculated and backpropogated
                    leaf_value = node.value
                else:
                    if self._init_critic_value:
                        leaf_value = node._initial_value
                    else:
                        leaf_value = reward_fn(simulate_env.get_state()).item()

        if done:
            node.set_as_terminate_node()
            if self.visited_paths is not None:
                self.visited_paths.append(
                    {
                        "text": simulate_env.answer,
                        "correct": winner == 1,
                        "value": leaf_value,
                    }
                )

        # Update value and visit count of nodes in this traversal.
        if simulate_env.mcts_mode == "play_with_bot_mode":
            node.update_recursive(leaf_value, simulate_env.mcts_mode)

        elif simulate_env.mcts_mode == "self_play_mode":
            # NOTE: e.g.
            #       to_play: 1  ---------->  2  ---------->  1  ----------> 2
            #         state: s1 ---------->  s2 ---------->  s3 ----------> s4
            #                                     action    node
            #                                            leaf_value
            # leaf_value is calculated from the perspective of player 1, leaf_value = value_func(s3),
            # but node.value should be the value of E[q(s2, action)], i.e. calculated from the perspective of player 2.
            # thus we add the negative when call update_recursive().
            node.update_recursive(-leaf_value, simulate_env.mcts_mode)

    def _select_child(
        self, node: LanguageNode, simulate_env: Type[CoTEnv]
    ) -> Tuple[Union[int, float], Node]:
        """
        Overview:
            Select the child with the highest UCB score.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        Returns:
            - action (:obj:`Int`): choose the action with the highest ucb score.
            - child (:obj:`Node`): the child node reached by executing the action with the highest ucb score.
        """

        def find_finish_reason(action):
            for legal_action in legal_actions:
                if legal_action["action"] == action:
                    return legal_action["finish_reason"]
        
        action = None
        child = None
        best_score = -9999999
        ucb_scores = []
        legal_actions = simulate_env.legal_actions
        finish_reaon = ""
        
        for action_tmp, child_tmp in node.children.items():
            ucb_score = self._ucb_score(node, child_tmp)
            score = ucb_score
            ucb_scores.append(ucb_score)
            if score > best_score:
                best_score = score
                action = action_tmp
                child = child_tmp
                finish_reaon = find_finish_reason(action_tmp)

        if child is None:
            child = node  # child==None, node is leaf node in play_with_bot_mode.

        return action, child , ucb_scores , finish_reaon

    def _select_by_prior(self, node: Node, simulate_env):
        data_tmp = [
            (x_action, x_node.prior_p) for x_action, x_node in node.children.items()
        ]
        action_list, prior_list = list(zip(*data_tmp))
        chosen_action = np.random.choice(action_list, p=np.array(prior_list))
        chosen_node = node.children[chosen_action]

        return chosen_action, chosen_node

    def _expand_leaf_node(
        self,
        node: Node,
        simulate_env: Type[CoTEnv],
        reward_fn: Optional[Callable] = None,
        depth: Optional[int] = None,
        target_length: Optional[int] = None,
    ) -> float:
        """
        Overview:
            expand the node with the reward_fn.
        Arguments:
            - node (:obj:`Class Node`): current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): the class of simulate env.
            - reward_fn (:obj:`Function`): the Callable to compute the state value.
        Returns:
            - leaf_value (:obj:`Bool`): the leaf node's value.
        """
        """
        action_probs_dict, leaf_value = reward_fn(simulate_env)
        for action, prior_p in action_probs_dict.items():
            if action in simulate_env.legal_actions:
                node.children[action] = Node(parent=node, prior_p=prior_p)
        """

        text_state = simulate_env.get_state()
        if not self._init_critic_value:
            leaf_value = reward_fn(text_state,self.root.depth)

        else:
            leaf_value = node._initial_value
            assert len(simulate_env.legal_actions) > 0
            prms = reward_fn(
                [
                    (
                        simulate_env.question,
                        simulate_env.answer + x["action"],
                    )
                    for x in simulate_env.legal_actions
                ],
                depth=depth,
                target_length=target_length,
            )
            child_values = []
            # PRM get last r as single reward
            for act, rs in zip(simulate_env.legal_actions, prms):
                if len(simulate_env.action_history) + 1 != len(rs):
                    '''
                    logger.warning(
                        "PRM value length not match with action history. \
                            len(prm)={}, len(act_hist)={} s:\n {}\n\na: \n{}\nrs:{}".format(
                            len(prms),
                            len(simulate_env.action_history),
                            text_state,
                            act,
                            rs,
                        )
                    )
                    # raise RuntimeError("Tokenizer problems")
                    '''
                    child_values.append(0.0)
                
                if len(rs) == 0:
                    '''
                    logger.warning(
                        "Empty PRM value for: \nState: \n{} \naction: \n{}, will be set to 0.0".format(
                            text_state, act
                        )
                    )
                    '''
                    child_values.append(0.0)
                else:
                    # prm-last
                    child_values.append(rs[-1])
                    # # prm-min
                    # child_values.append(min(rs))
                    # # prob-prm
                    # child_values.append(act['prob'])

        assert len(node.children) == 0
        for i, action_dict in enumerate(simulate_env.legal_actions):
            action, prob = action_dict["action"], action_dict["prob"]

            if self._init_critic_value:
                child_value = child_values[i]
            else:
                # XXX(ziyu): consider turn off this branch, i.e. always assume
                #  `self._init_critic=True`, since with LLM
                child_value = 0.0
            
            node.children[action] = LanguageNode(
                parent=node,
                prior_p=prob,
                #  prm_value=prm_value,
                text_state=text_state,
                last_action=action,
                initial_value=child_value,
                num_generated_token=action_dict["num_token"],
            )
            # set terminal node here
            if simulate_env._next_state_terminated[action]:
                node.children[action].set_as_terminate_node()
        if len(node.children) == 0:
            print_rank_0(
                "Prune all current children at node {}".format(node.last_action)
            )
        
        # collect num tokens
        if not node.has_collected_token_num:
            self._completion_tokens += sum(
                c.num_generated_token for c in node.children.values()
            )
            node.has_collected_token_num = True
        else:
            raise RuntimeError("Token number has been collected again.")

        return leaf_value

    def _ucb_score(self, parent: Node, child: Node) -> float:
        """
        Overview:
            Compute UCB score. The score for a node is based on its value, plus an exploration bonus based on the prior.
        Arguments:
            - parent (:obj:`Class Node`): Current node.
            - child (:obj:`Class Node`): Current node's child.
        Returns:
            - score (:obj:`Bool`): The UCB score.
        """
        pb_c = (
            math.log((parent.visit_count + self._pb_c_base + 1) / self._pb_c_base)
            + self._pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior_p
        value_score = child.value

        return prior_score + value_score
        # return value_score

    def reset_prior(self, node: Node) -> None:
        """
        Overview:
            Reset prior probability
        Arguments:
            - node (:obj:`Class Node`): Current node.
        """
        for a in node.children.keys():
            node.children[a].prior_p = node.children[a].prior_p_ori

    def _add_exploration_noise(self, node: Node) -> None:
        """
        Overview:
            Add exploration noise.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        """
        # Get a list of actions corresponding to the child nodes.
        actions = list(node.children.keys())
        # Create a list of alpha values for Dirichlet noise.
        alpha = [self._root_dirichlet_alpha] * len(actions)
        # Generate Dirichlet noise using the alpha values.
        noise = np.random.dirichlet(alpha)
        # Compute the weight of the exploration noise.
        frac = self._root_noise_weight
        # Update the prior probability of each child node with the exploration noise.
        for a, n in zip(actions, noise):
            node.children[a].prior_p = node.children[a].prior_p * (1 - frac) + n * frac

    @classmethod
    def from_json(cls, cfg: dict, json_path: str, reset_visit_info: bool):
        tree_json = json.load(open(json_path, "r"))

        def build_tree(tree_dict: dict) -> Node:
            node_info = tree_dict["info"]
            current_node = LanguageNode(
                text_state=node_info.get("text_state", None),
                last_action=node_info.get("last_action", None),
                prior_p=node_info["prior_p"],
                prm_value=node_info.get("prm_value", None),
                initial_value=node_info.get("initial_value", 0.0),
            )

            if not reset_visit_info:
                current_node._visit_count = node_info["visit_cnt"]
                current_node._value_sum = node_info["value"] * current_node.visit_count
            if node_info.get("terminated", False):
                current_node.set_as_terminate_node()

            for name, child_dict in tree_dict["children"].items():
                child_node = build_tree(child_dict)
                current_node._children[name] = child_node
                child_node._parent = current_node

            return current_node

        root_node = build_tree(tree_dict=tree_json)

        obj = cls(cfg)
        obj.root = root_node
        return obj
    
    def draw_tree(self):
        # Not tested yet
        root = self.root
        assert root, 'Root node is None'
        def draw_node(node, depth):
            print('|' + '-' * depth + str(node))
            for child in node.children.values():
                draw_node(child, depth + 1)
        
        print(f"\n---------Expanded Tree---------")
        draw_node(self.root)
