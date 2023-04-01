config = None
cur_seed = None
cur_time_step_iter = None
always_save_iter = [24, 25, 26]
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
        name += str(t) + "_" + to_str(globals()[t]) + "_"
    return name

def to_str(t):
    if type(t) is int:
        return "{:02d}".format(t)
    else:
        return str(t)

class TurnOffRequiresGradDeepLatent(object):     
    def __enter__(self):
        global deepLatentRequiresGrad
        deepLatentRequiresGrad = False
        return deepLatentRequiresGrad
 
    def __exit__(self, *args):
        global deepLatentRequiresGrad
        deepLatentRequiresGrad = True
 