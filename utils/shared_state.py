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

#hyperParamDefaults = {"strict":False, "inside_loss_scale":.2, "outside_loss_scale":.2,"shrink_factor":0, "thresholds": {0: 0.2, 4: 0.8, 5: .9, 6: .9, 8: .9, 12: .9}, "meta_prompt": "a [robot:.6,.3,.4,.55] and a [blue vase:0,.3,.4,.55]"}
hyperParamDefaults = {"strict":False, "inside_loss_scale":.2, "outside_loss_scale":.2,"shrink_factor":.15, "thresholds": {0:.4, 2:.8, 4:.9, 8:.9}, "meta_prompt": "a [robot:.6,.3,.4,.55] and a [vase:0,.3,.4,.55] and the [moon:.35,.05,.35,.35]"}
hyperParamOverrides = [{}]#{"thresholds": {0: 0.1, 15: 0.8}}, {"thresholds": {0: 0.1, 20: 0.8}}, {"thresholds": {0: 0.1, 30: 0.8}}, {"thresholds": {0: 0.1, 40: 0.8}}]

def get_hyperparam_states():
    hyperParamStateDicts = []
    for overrides in hyperParamOverrides:
        new_state = hyperParamDefaults.copy()
        for override in overrides:
            new_state[override] = overrides[override]
        hyperParamStateDicts.append(new_state)
    return hyperParamStateDicts


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
 