import pprint
from typing import List
from abc import ABC, abstractmethod

import pyrallis
import torch
from PIL import Image

from config import RunConfig
from pipeline_guided_attention import GuidedAttention
from utils import ptp_utils, vis_utils, shared_state, helpers
from utils.ptp_utils import AttentionStore

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_model(config: RunConfig):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if config.sd_2_1:
        stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
    revision = None
    if config.half_precision:
        revision = "fp16"
    stable = GuidedAttention.from_pretrained(stable_diffusion_version, revision=revision).to(device)
    return stable


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices


def run_on_prompt(prompt: List[str],
                  model: GuidedAttention,
                  controller: AttentionStore,
                  seed: torch.Generator,
                  config: RunConfig) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs = model(prompt=prompt,
                    attention_store=controller,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1)
    image = outputs.images[0]
    return image

def get_indices(tokenized_prompt, tokens):
    tokens_len = len(tokens)
    for i in range(0,len(tokenized_prompt) - tokens_len):
        if tokenized_prompt[i:i+tokens_len] == tokens:
            return list(range(i,i+tokens_len))

def overrideConfig(config):
    if 'meta_prompt' in shared_state.curHyperParams:
        config.meta_prompt = shared_state.curHyperParams['meta_prompt']
    if 'thresholds' in shared_state.curHyperParams:
        config.thresholds = shared_state.curHyperParams["thresholds"]

def parseMetaPrompt(config):
    config.prompt, config.meta_info, config.custom_loss = helpers.parse_prompt(config.meta_prompt)
    shared_state.config = config
    tokenized_prompt = config.stable.tokenizer(config.prompt)['input_ids']
    token_dict = {} #relates indice to word and loss
    for meta_info_item in config.meta_info:
        tokens = config.stable.tokenizer(meta_info_item[0])['input_ids'][1:-1]
        indices = get_indices(tokenized_prompt, tokens)
        for indice in indices:
            token_dict[indice] = {'word':config.stable.tokenizer.decode(tokenized_prompt[indice]), 'loss_type' : meta_info_item[1], 'loss' : meta_info_item[2], 'subprompt' : meta_info_item[0]}
    config.token_dict = token_dict

def execute(config):

    images = []
    image_path = None
    for seed in config.seeds:
        for hyperParamState in shared_state.get_hyperparam_states():
            shared_state.curHyperParams = hyperParamState
            overrideConfig(config)
            parseMetaPrompt(config)
            helpers.log_clear()
            
            shared_state.cur_seed = seed
            print(f"Seed: {seed}")
            g = torch.Generator('cuda').manual_seed(seed)
            controller = AttentionStore()
            image = run_on_prompt(prompt=config.prompt,
                                model=config.stable,
                                controller=controller,
                                seed=g,
                                config=config)
            prompt_output_path = config.output_path / helpers.get_inner_folder_name()
            prompt_output_path.mkdir(exist_ok=True, parents=True)
            
            helpers.annotate_image(image)
            name1 = helpers.dictToString(shared_state.curHyperParams)
            image_path = prompt_output_path / f'{seed}{name1}.png'
            try:
                image.save(image_path)
            except:
                print("bad path. this is often due to exceeding max path length.")
                name1 = ""
                image_path = prompt_output_path / f'{seed}{name1}.png'
                image.save(image_path)
            helpers.log_save(prompt_output_path / f'{seed}{name1}.txt')
            helpers.save_latent_stats(prompt_output_path / f'{seed}{name1}figure.png')
            images.append(image)

    # save a grid of results across all seeds
    joined_image = vis_utils.get_image_grid(images)
    if not config.interactive:
        helpers.annotate_image(joined_image)
    joined_image.save(config.output_path / f'{helpers.get_meta_prompt_clean()}.png')
    return image_path
    
skipLoading = False

def setup(config):
    shared_state.config = config
    if not skipLoading:
        if config.half_precision:
            torch.set_default_tensor_type(torch.HalfTensor)
        stable = load_model(config)
        config.stable = stable


class CustomLossBase(ABC):
    @abstractmethod
    def calc_loss(self, cross_attention_maps, text_args : str) -> torch.Tensor:
        pass

    # optional method. used for diagnostic purposes.
    def subprompts_of_interest(self, text_args : str) -> list[str]:
        return []

    # convenience methods
    def parse_text_args(self, text_args : str):
        import ast
        return ast.literal_eval(text_args)
    
    def find_indices_for_sub_prompt(self, sub_prompt):
        full_prompt = shared_state.config.stable.tokenizer(shared_state.config.prompt)['input_ids'][1:-1]
        sub_prompt = shared_state.config.stable.tokenizer(sub_prompt)['input_ids'][1:-1]
        for i in range(len(full_prompt) - len(sub_prompt) + 1):
            if full_prompt[i:i + len(sub_prompt)] == sub_prompt:
                return list(range(i, i + len(sub_prompt)))
    
    def get_map_for_token(self, cross_attention_maps, token_index : int, pixel_wise_normalization : True):
        image_map = cross_attention_maps[:, :, token_index]
        if pixel_wise_normalization:
            image_map = image_map / image_map.sum()
        return image_map

        

class ToLeftOf(CustomLossBase):
    
    def calc_loss(self, cross_attention_maps, text_args : str) -> torch.Tensor:
        # goes from text args i.e. (cat, vase) to tuple of strings i.e. ("cat", "vase").
        text_args = self.quote_items_in_tuple(text_args)
        args = self.parse_text_args(text_args)

        # get token indices for sub prompt (i.e. "cat" -> 2).
        left_side_indices = self.find_indices_for_sub_prompt(args[0])
        right_side_indices = self.find_indices_for_sub_prompt(args[1])

        # calculate center x for the two subprompts.
        left_weighted_center = torch.Tensor([0.]).cuda()
        right_weighted_center = torch.Tensor([0.]).cuda()
        for i in left_side_indices:
            map = self.get_map_for_token(cross_attention_maps, i, True)
            left_weighted_center += self.calc_weighted_center(map)[0] / len(left_side_indices)
        for i in right_side_indices:
            map = self.get_map_for_token(cross_attention_maps, i, True)
            right_weighted_center += self.calc_weighted_center(map)[0] / len(left_side_indices)

        # calculate loss. loss is non zero as long as gap of 20% of map width is between left and right objects.
        map_width = cross_attention_maps.shape[1] # H x W x NumTokens
        gap = .2 * map_width
        loss = (left_weighted_center + torch.Tensor([gap]).cuda() - right_weighted_center) / map_width
        loss *= 9 # custom weight
        return torch.max(loss, torch.Tensor([0]).cuda())
    
    def subprompts_of_interest(self, text_args : str) -> list[str]:
        text_args = self.quote_items_in_tuple(text_args)
        args = self.parse_text_args(text_args)
        return list(args)
        
    def quote_items_in_tuple(self, text_args):
        items = text_args.strip('()').split(',')
        quoted_items = [f"'{item.strip()}'" for item in items]
        quoted_input_str = f"({','.join(quoted_items)})"
        return quoted_input_str


    def calc_weighted_center(self, imageNormalized):
        weighted_center_col = torch.Tensor([0.]).cuda()
        weighted_center_row = torch.Tensor([0.]).cuda()
        for ii in range(0, 16):
            for jj in range(0, 16):
                # sample pixels at center. so center of image is (8,8). without adding .5 it would be at (7.5 7.5).
                weighted_center_col += (jj + .5) * imageNormalized[ii][jj] #weighed x. 
                weighted_center_row += (ii + .5) * imageNormalized[ii][jj] #weighed y.
        return weighted_center_col, weighted_center_row
        


def register_custom_loss(name : str, customLoss : CustomLossBase):
    if not hasattr(shared_state.config,"registered_loss_functions"):
        shared_state.config.registered_loss_functions = {}
    shared_state.config.registered_loss_functions[name] = customLoss


@pyrallis.wrap()
def main(config: RunConfig):
    setup(config)

    # this is one place to register custom loss functions
    register_custom_loss("toLeftOf", ToLeftOf())

    if config.interactive:
        import gui
        gui.run()
    else:
        execute(config)



if __name__ == '__main__':
    main()
