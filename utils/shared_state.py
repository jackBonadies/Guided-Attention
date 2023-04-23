config = None
cur_seed = None
cur_time_step_iter = None
always_save_iter = [24, 25, 26]
sub_iteration = 0

sigmas = None #DDIM / LDMS sigmas
timesteps = None #interpolated training timesteps i.e. 981, 961, ... 1

optimizeDeepLatent = False
use_loss_total = True

deepLatentRequiresGrad = True
injectDeepFeatures = False
deepFeatures = None

curHyperParams = None

#hyperParamDefaults = {"strict":False, "inside_loss_scale":.2, "outside_loss_scale":.2,"shrink_factor":0, "thresholds": {0: 0.2, 4: 0.8, 5: .9, 6: .9, 8: .9, 12: .9}, "meta_prompt": "a [robot:.6,.3,.4,.55] and a [blue vase:0,.3,.4,.55]"}
#hyperParamDefaults = {"strict":False, "inside_loss_scale":.2, "outside_loss_scale":.2,"shrink_factor":.15, "thresholds": {0:.4, 2:.8, 4:.9, 8:.9}, "meta_prompt": "a [robot:.6,.3,.4,.55] and a [vase:0,.3,.4,.55] and the [moon:.35,.05,.35,.35]"}
hyperParameterOverrides = {"strict":False, "inside_loss_scale":.2, "outside_loss_scale":.2,"shrink_factor":.15, "thresholds": {0:1.}, "use_optimizer":False,"recurse_until":14, "recurse_steps":3} #, "meta_prompt": "a [robot:.6,.3,.4,.55] and a [antique vase:.2,.3,.4,.55]"
hyperParameterIterations = [{}]#{"thresholds": {0: 0.1, 15: 0.8}}, {"thresholds": {0: 0.1, 20: 0.8}}, {"thresholds": {0: 0.1, 30: 0.8}}, {"thresholds": {0: 0.1, 40: 0.8}}]
#, 1:1., 2:1., 3:1., 4:1., 5:1., 6:1., 7:1., 8:1., 9:1., 10:1., 11:1., 12:1., 13:1.


def get_sigma():
    return sigmas[timesteps[cur_time_step_iter]]

def get_hyperparam_states():
    hyperParamStateDicts = []
    for overrides in hyperParameterIterations:
        new_state = hyperParameterOverrides.copy()
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
 