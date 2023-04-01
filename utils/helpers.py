import utils.shared_state as state
from PIL import Image, ImageDraw, ImageFont
from enum import Enum
import numpy as np
import math
import torch as tr

class AnnotationType(Enum):
    COOR = 0
    BOX = 1
    KEYWORD = 2

class Rect():
    def __init__(self, x, y, width, height, size):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.size = size
    def right(self):
        return self.x + self.width
    def bottom(self):
        return self.y + self.height
    def center(self):
        return ((self.x + self.width / 2.0), (self.y + self.height / 2.0))
    def of_size(self, new_size):
        ratio = float(new_size/self.size)
        return Rect(self.x*ratio, self.y*ratio, self.width*ratio, self.height*ratio, new_size)


def add_word(prompt, token):
    if len(prompt) == 0 or prompt[-1] == ' ':
        prompt += token
    else:
        prompt += ' ' + token
    return prompt

def parse_prompt(meta_prompt):
    cur_index = 0
    prompt = ''
    meta_info = []
    while(True):
        meta_prompt = meta_prompt.lstrip(' ')
        space_index = -1
        if ' ' in meta_prompt:
            space_index = meta_prompt.index(' ')
        meta_index = -1
        if '[' in meta_prompt:
            meta_index = meta_prompt.index('[')
        if space_index == -1 and meta_index == -1:
            return (prompt, meta_info)
        if meta_index == -1:
            return (add_word(prompt,meta_prompt[0:]), meta_info)
        if space_index == -1 or meta_index < space_index:
            end_meta_index = meta_prompt.index(']')
            colon_index = meta_prompt.index(':')
            token = meta_prompt[meta_index+1:colon_index].strip(' ')
            coors = meta_prompt[colon_index+1:end_meta_index].strip(' ')
            numbers = coors.split(',')
            if(len(numbers) == 2):
                x = float(numbers[0])
                y = float(numbers[1])
                meta_info.append((token, AnnotationType.COOR, (x,y)))
            elif(len(numbers) == 4):
                x = float(numbers[0])
                y = float(numbers[1])
                width = float(numbers[2])
                height = float(numbers[3])
                meta_info.append((token, AnnotationType.BOX, Rect(x, y, width, height, 1)))
            else:
                pass
            prompt = add_word(prompt,token)
            meta_prompt = meta_prompt[end_meta_index + 1:]

            #meta_prompt.index()
        else: # normal token
            token = meta_prompt[0:space_index + 1]
            prompt = add_word(prompt,token)
            meta_prompt = meta_prompt[space_index:]
    return prompt, meta_info


def get_meta_prompt_clean():
    return state.config.meta_prompt.replace('[','_').replace(']','_').replace(':','_').replace('.','_')

def annotate_image(image):
    if state.config.annotate:
        draw  = ImageDraw.Draw(image)
        font  = ImageFont.truetype("arial.ttf", 20, encoding="unic")
        for m_info in state.config.meta_info:
            word = m_info[0]
            if m_info[1] == AnnotationType.COOR:
                xy = m_info[2]
                x = xy[0] * 16
                y = xy[1] * 16
                x_desired = x
                y_desired = y
                length = 15
                draw.line( [((512)/16*(x_desired) - length,512/16*y_desired),((512)/16*(x_desired) + length,512/16*y_desired)], fill="#a00000" )
                draw.line( [(512)/16*(x_desired),(512)/16*y_desired - length,(512)/16*(x_desired),(512)/16*y_desired + length], fill="#a00000" )
                draw.text( (512/16*x,512/16*y), word, fill="#a00000", font=font)
            elif m_info[1] == AnnotationType.BOX:
                rect = m_info[2]
                shape = [(rect.x * 512, rect.y * 512), (rect.right() * 512, rect.bottom() * 512)]
                draw.rectangle(shape, fill=None, width=2, outline="#a00000")
                draw.text( (512*rect.x,512*rect.y), word, fill="#a00000", font=font)

#bounding box helper functions...
sample_center = True
shrink_box = True

def get_corresponding_weight(x):
    xp = [0, .333, .666, 1.0]
    fp = [3, 2.5, 1, .2] #hard drop off near edges
    return np.interp(x, xp, fp)


def inside_box(cur_x, cur_y, rect):
    if sample_center:
        cur_x += 0.5
        cur_y += 0.5
    offsetX = state.curHyperParams["shrink_factor"] * rect.width
    offsetY = state.curHyperParams["shrink_factor"] * rect.height
    if cur_x >= (rect.x + offsetX) and cur_x <= (rect.x + rect.width - offsetX):
        if cur_y >= (rect.y + offsetY) and cur_y <= (rect.y + rect.height - offsetY):
            return True
    return False

def distance_from_center(cur_x, cur_y, rect, normalized):
    if sample_center:
        cur_x += 0.5
        cur_y += 0.5
    if normalized: #each dim separately
        dist = math.sqrt(math.pow(2*(rect.center()[0] - cur_x)/rect.width, 2) + math.pow(2*(rect.center()[1] - cur_y)/rect.height, 2)) / math.sqrt(2)
    else:
        dist = math.sqrt(math.pow(rect.center()[0] - cur_x, 2) + math.pow(rect.center()[1] - cur_y, 2))
        # if normalized: #overall
        #     dist = dist / (r.diag() /2.0)
    return dist


def distance_from_bounding_box(cur_x, cur_y, rect, normalized):
    if sample_center:
        cur_x += 0.5
        cur_y += 0.5
    if normalized:
        raise NotImplementedError()
    else:
        dist_x = 0
        if cur_x < rect.x:
            dist_x = abs(cur_x - rect.x)
        elif cur_x > rect.right():
            dist_x = abs(cur_x - rect.right())
        dist_y = 0
        if cur_y < rect.y:
            dist_y = abs(cur_y - rect.y)
        elif cur_y > rect.bottom():
            dist_y = abs(cur_y - rect.bottom())
    return dist_x + dist_y


def get_corresponding_weight_distance_from(dist):
    return 1.0
    # if dist == 1:
    #     return .5
    # else:
    #     return 1.0

def calculate_bounding_box_losses(r, imageSoftmax):
    weights = tr.ones(16,16).cuda()
    for ii in range(0, 16):
        for jj in range(0, 16):
            if inside_box(jj, ii, r):
                c_dist = distance_from_center(jj, ii, r, True) #0 == at center. 1 == at furthest corner.
                w1 = get_corresponding_weight(c_dist) #not normalized
                weights[ii,jj] = w1
            else:
                dist = distance_from_bounding_box(jj, ii, r, False)
                w1 = get_corresponding_weight_distance_from(dist) #not normalized
                weights[ii,jj] = w1
    num_inside = 0
    sum_inside = 0
    num_outside = 0
    sum_outside = 0
    for ii in range(0, 16):
        for jj in range(0, 16):
            if inside_box(jj, ii, r):
                sum_inside += weights[ii,jj]
                num_inside += 1
            else:
                sum_outside += weights[ii,jj]
                num_outside += 1

    # normalized for bounding box size            
    for ii in range(0, 16):
        for jj in range(0, 16):
            if inside_box(jj, ii, r):
                weights[ii,jj] /= sum_inside
            else:
                weights[ii,jj] /= sum_outside
    loss_inside = tr.Tensor([0]).cuda()
    loss_outside = tr.Tensor([0]).cuda()

    if state.curHyperParams["strict"]:
        # loss
        at_most = 1.0 / num_inside #since these are softmaxed. cannot expect every value inside a 10x10 region to be above .5 say.. at most they could each be .01
        
        min_loss = tr.Tensor([0]).cuda()
        min_attn = tr.Tensor([0]).cuda()
        for ii in range(0, 16):
            for jj in range(0, 16):
                if inside_box(jj, ii, r):
                    loss_item = 2.*max(min_loss, at_most-imageSoftmax[ii, jj])
                    loss_inside += weights[ii,jj]*loss_item
                else:
                    loss_item = max(min_loss, imageSoftmax[ii, jj] - min_attn) #no problem if less than .02
                    loss_outside += weights[ii,jj]*loss_item
        return (loss_inside, loss_outside)
    else:
        at_most = 1.0 / num_inside 
        attn_sum_inside = tr.Tensor([0]).cuda()
        attn_sum_outside = tr.Tensor([0]).cuda()
        for ii in range(0, 16):
            for jj in range(0, 16):
                if inside_box(jj, ii, r):
                    attn_sum_inside += imageSoftmax[ii, jj]
                else:
                    attn_sum_outside += imageSoftmax[ii, jj]
        loss_inside = 1. - attn_sum_inside # we want inside to be 1
        loss_outside = attn_sum_outside # we want outside to be 0
        return (loss_inside, loss_outside)



def dictToString(dict1):
    if type(dict1) is dict:
        str1 = ""
        for item in dict1.items():
            str1 += "_" + str(item[0]) + "_" + dictToString(item[1])
        return str1
    else:
        return str(dict1)
