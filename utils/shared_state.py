config = None
cur_seed = None
cur_time_step_iter = None
sub_iteration = 0

toCoordinate = True
coor_X = 3.
coor_Y = 4.
toRight = True
optimizeDeepLatent = False

deepLatentRequiresGrad = True
injectDeepFeatures = False
deepFeatures = None

tags = ["cur_seed", "cur_time_step_iter", "optimizeDeepLatent"]

def get_name():
    name = ""
    for t in tags:
        name += str(t) + "_" + str(globals()[t]) + "_"
    if toCoordinate:
        name += "toCoordinate_X_" + str(coor_X) + "_Y_" + str(coor_Y) + "_"
    else:
        name += "toRight_" + str(toRight) + "_"
    return name

class TurnOffRequiresGradDeepLatent(object):     
    def __enter__(self):
        global deepLatentRequiresGrad
        deepLatentRequiresGrad = False
        return deepLatentRequiresGrad
 
    def __exit__(self, *args):
        global deepLatentRequiresGrad
        deepLatentRequiresGrad = True
 