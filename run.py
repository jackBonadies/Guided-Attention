import pprint
from typing import List

import pyrallis
import torch
from PIL import Image

from config import RunConfig
from pipeline_attend_and_excite import AttendAndExcitePipeline
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
    stable = AttendAndExcitePipeline.from_pretrained(stable_diffusion_version, revision=revision).to(device)
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
                  model: AttendAndExcitePipeline,
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

def execute(config):
    config.prompt, config.meta_info = helpers.parse_prompt(config.meta_prompt)
    shared_state.config = config
    tokenized_prompt = config.stable.tokenizer(config.prompt)['input_ids']
    token_dict = {} #relates indice to word and loss
    for meta_info_item in config.meta_info:
        tokens = config.stable.tokenizer(meta_info_item[0])['input_ids'][1:-1]
        indices = get_indices(tokenized_prompt, tokens)
        for indice in indices:
            token_dict[indice] = {'word':config.stable.tokenizer.decode(tokenized_prompt[indice]), 'loss_type' : meta_info_item[1], 'loss':meta_info_item[2]}
    config.token_dict = token_dict
    images = []
    image_path = None
    for seed in config.seeds:
        shared_state.cur_seed = seed
        print(f"Seed: {seed}")
        g = torch.Generator('cuda').manual_seed(seed)
        controller = AttentionStore()
        image = run_on_prompt(prompt=config.prompt,
                            model=config.stable,
                            controller=controller,
                            seed=g,
                            config=config)
        prompt_output_path = config.output_path / config.prompt
        prompt_output_path.mkdir(exist_ok=True, parents=True)
        image_path = prompt_output_path / f'{seed}.png'
        helpers.annotate_image(image)
        image.save(prompt_output_path / f'{seed}.png')
        images.append(image)

    # save a grid of results across all seeds
    joined_image = vis_utils.get_image_grid(images)
    helpers.annotate_image(joined_image)
    joined_image.save(config.output_path / f'{helpers.get_meta_prompt_clean()}.png')
    return image_path
    
skipLoading = False

@pyrallis.wrap()
def main(config: RunConfig):
    shared_state.config = config
    if not skipLoading:
        if config.half_precision:
            torch.set_default_tensor_type(torch.HalfTensor)
        stable = load_model(config)
        config.stable = stable
    if config.interactive:
        import gui
        gui.run()
    else:
        execute(config)



if __name__ == '__main__':
    main()
