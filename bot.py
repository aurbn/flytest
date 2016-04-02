import interface as bbox
import numpy as np


def get_action_by_state(state):
	best_act = -1
	best_val = -1e9
 
	for act in xrange(n_actions):
		val = calc_reg_for_action(act, state)
		if val > best_val:
			best_val = val
			best_act = act
 
	return best_act


 
n_features = n_actions = -1
 
 
def prepare_bbox():
	global n_features, n_actions
 
	if bbox.is_level_loaded():
		bbox.reset_level()
	else:
		bbox.load_level("train_level.data", verbose=1)
		n_features = bbox.get_num_of_features()
		n_actions = bbox.get_num_of_actions()
 
 
def load_regression_coefs(filename):
	global reg_coefs, free_coefs
	coefs = np.loadtxt(filename).reshape(n_actions, n_features + 1)
	reg_coefs = coefs[:,:-1]
	free_coefs = coefs[:,-1]
 
 
def calc_reg_for_action(action, state):
	return np.dot(reg_coefs[action], state) + free_coefs[action]
 
 
def run_bbox(verbose=False):
	has_next = 1
	
	prepare_bbox()
	load_regression_coefs("reg_coefs.txt")
 
	while has_next:
		state = bbox.get_state()
		action = get_action_by_state(state)
		#print bbox.get_score()
		has_next = bbox.do_action(action)
 
	bbox.finish(verbose=1)
 
 
if __name__ == "__main__":
	run_bbox()
