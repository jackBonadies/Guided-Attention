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

curHyperParams = None
hyperParameterDicts = [{"strict":False, "inside_loss_scale":.2, "outside_loss_scale":.2,"shrink_factor":.25, "thresholds": {0: 0.1, 2: 0.8}},{"strict":False, "inside_loss_scale":.2, "outside_loss_scale":.2,"shrink_factor":.25, "thresholds": {0: 0.1, 4: 0.8}},{"strict":False, "inside_loss_scale":.2, "outside_loss_scale":.2,"shrink_factor":.25, "thresholds": {0: 0.1, 5: 0.8}},{"strict":False, "inside_loss_scale":.2, "outside_loss_scale":.2,"shrink_factor":.25, "thresholds": {0: 0.1, 6: 0.8}}]

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
 