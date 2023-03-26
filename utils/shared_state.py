config = None
cur_seed = None
cur_time_step_iter = None
sub_iteration = 0

optimizeDeepLatent = False
use_loss_total = True

deepLatentRequiresGrad = True
injectDeepFeatures = False
deepFeatures = None

tags = ["cur_seed", "cur_time_step_iter", "optimizeDeepLatent"]

def get_name():
    name = ""
    for t in tags:
        name += str(t) + "_" + str(globals()[t]) + "_"
    return name

class TurnOffRequiresGradDeepLatent(object):     
    def __enter__(self):
        global deepLatentRequiresGrad
        deepLatentRequiresGrad = False
        return deepLatentRequiresGrad
 
    def __exit__(self, *args):
        global deepLatentRequiresGrad
        deepLatentRequiresGrad = True
 