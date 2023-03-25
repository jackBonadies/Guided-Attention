import utils.shared_state as state
from PIL import Image, ImageDraw, ImageFont

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
                meta_info.append((token, "Coordinate", (x,y)))
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
            xy = m_info[2]
            x = xy[0] * 16
            y = xy[1] * 16
            x_desired = x
            y_desired = y
            length = 15
            draw.line( [((512)/16*(x_desired) - length,512/16*y_desired),((512)/16*(x_desired) + length,512/16*y_desired)], fill="#a00000" )
            draw.line( [(512)/16*(x_desired),(512)/16*y_desired - length,(512)/16*(x_desired),(512)/16*y_desired + length], fill="#a00000" )
            draw.text( (512/16*x,512/16*y), word, fill="#a00000", font=font)