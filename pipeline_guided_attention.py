
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch

import matplotlib.pyplot as plt

from torch.nn import functional as F

from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, is_accelerate_available, logging, randn_tensor, replace_example_docstring
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

from diffusers.models import unet_2d_condition

from diffusers import DDIMScheduler

import utils.shared_state as state
from utils import helpers

from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

from utils.gaussian_smoothing import GaussianSmoothing
from utils.ptp_utils import AttentionStore, aggregate_attention

logger = logging.get_logger(__name__)

class AttendAndExcitePipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return text_inputs, prompt_embeds

    def _compute_max_attention_per_index(self,
                                         attention_maps: torch.Tensor,
                                         smooth_attentions: bool = False,
                                         sigma: float = 0.5,
                                         kernel_size: int = 3,
                                         normalize_eot: bool = False) -> List[torch.Tensor]:
        losses_dict = {}
        """ Computes the maximum attention value for each of the tokens we wish to alter. """
        last_idx = -1
        if normalize_eot:
            prompt = self.prompt
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        #note that indexing a tensor does NOT make a copy.  so the actions on attention_for_text
        #  do modify attention_maps
        attention_for_text = attention_maps[:, :, 1:last_idx] #these are softmax (say .07 max and .0001 min) (including token 0 max is .8843)
        attention_for_text *= 100 #multiply by 100 and take softmax again gives .8866 max and .001 min
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)
        # ^ now per pixel we get the softmax attn values NOT including the global token

        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in state.config.token_dict.keys()]

        # Extract the maximum values
        max_indices_list = []
        col_list = []
        row_list = []
        inside_loss_list = []
        outside_loss_list = []
        for i in indices_to_alter:
            image = attention_for_text[:, :, i] #16, 16
            self.save_viridis(image, "_attnmap_" + self.get_token(i + 1))
            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0) #max gonna be a bit lower
            max_indices_list.append(image.max())

            weighted_center_col = torch.Tensor([0.]).cuda()
            weighted_center_row = torch.Tensor([0.]).cuda()

            # this is softmax for pixel space (i.e. which pixels pay most attn to token x i.e. attn[:,:,0].sum() == 1) 
            # whereas above was which tokens does pixel y pay most attn. (i.e. attn[0][0].sum() == 1)
            #imageSoftmax = torch.nn.functional.softmax(image.flatten() * .2, dim=-1).reshape(16,16)
            imageSoftmax = image / image.sum()
            for ii in range(0, 16):
                for jj in range(0, 16):
                    # sample pixels at center. so center of image is (8,8). without adding .5 it would be at (7.5 7.5).
                    weighted_center_col += (jj + .5) * imageSoftmax[ii][jj] #weighed x. 
                    weighted_center_row += (ii + .5) * imageSoftmax[ii][jj] #weighed y.

            #indexMax = image.flatten().argmax() #NOT differentiable...
            # col = indexMax % 16
            # row = int(indexMax / 16)
            print("weighted center col: " + str(weighted_center_col.item()))
            print("weighted center row: " + str(weighted_center_row.item()))
            col_list.append(weighted_center_col)
            row_list.append(weighted_center_row)
            if state.config.token_dict[i+1]['loss_type'] == helpers.AnnotationType.BOX:
                rect = state.config.token_dict[i+1]['loss']
                inside_loss, outside_loss = helpers.calculate_bounding_box_losses(rect.of_size(16.0), imageSoftmax)
                inside_loss_list.append(inside_loss)
                outside_loss_list.append(outside_loss)
            else:
                inside_loss_list.append(0)
                outside_loss_list.append(0)

        losses_dict["max_loss"] = max_indices_list #list is in order.
        losses_dict["col"] = col_list
        losses_dict["row"] = row_list
        losses_dict["inside_loss"] = inside_loss_list
        losses_dict["outside_loss"] = outside_loss_list
        return losses_dict

    def _aggregate_and_get_max_attention_per_token(self, attention_store: AttentionStore,
                                                   attention_res: int = 16,
                                                   smooth_attentions: bool = False,
                                                   sigma: float = 0.5,
                                                   kernel_size: int = 3,
                                                   normalize_eot: bool = False):
        """ Aggregates the attention for each token and computes the max activation value for each token to alter. """
        from_where=("up", "down", "mid")
        if state.optimizeDeepLatent:
            from_where=("up", ) # we have no effect on the down side of things.
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=from_where,
            is_cross=True,
            select=0)
        losses_dict = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
        return losses_dict

    @staticmethod
    def _compute_loss(losses_dict: dict[str,List[torch.Tensor]], return_losses: bool = False) -> torch.Tensor:
        """ Computes the attend-and-excite loss using the maximum attention value for each token. """
        losses = []
        for i in range(0, len(losses_dict["max_loss"])):
            first_index = list(state.config.token_dict.keys())[i]
            token_info = state.config.token_dict[first_index]
            
            #losses_dict["max_loss"][i]
            if token_info['loss_type'] == helpers.AnnotationType.COOR:
                xy = token_info['loss']
                part1 = max(0., 1.*(losses_dict["col"][i] - xy[0]*16).abs()/15.) #8.* caused serious artifacts when optimizing in pixel space
                part2 = max(0., 4.*(losses_dict["row"][i] - xy[1]*16).abs()/15.) #1. -- way too weak...
                losses.append(part1 + part2)
            elif token_info['loss_type'] == helpers.AnnotationType.BOX:
                rect = token_info['loss']
                center = rect.center()
                # should be centered
                # part1 = max(0., 1.*(losses_dict["col"][i] - center[0]*16).abs()/15.) #8.* caused serious artifacts when optimizing in pixel space
                # part2 = max(0., 4.*(losses_dict["row"][i] - center[1]*16).abs()/15.) #1. -- way too weak...

                part3 = 10.*losses_dict["inside_loss"][i] 

                part4 = 100.*losses_dict["outside_loss"][i] 

                losses.append(part3 + part4)
            # elif state.toRight:
            #     losses.append(max(0, 2*(15. - losses_dict["col"][i])/15.)) # loss for each token
            # else:
            #     losses.append(max(0, 2*(losses_dict["col"][i])/15.)) # loss for each token

        loss = max(losses) # we only care about the token with max loss (TODO: potential optimization - we should care about all losses that are above the threshold)
        if return_losses:
            return loss, losses
        else:
            return loss

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float) -> torch.Tensor:
        """ Update the latent according to the computed loss. """
        if state.optimizeDeepLatent:
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [state.deepFeatures], retain_graph=True)[0]
            print("gradient size: " + str(grad_cond.abs().mean().item()))
            print("gradient size: " + str(grad_cond.abs().sum().item()))
            # 3000, 100 produce similar results. both rather low quality. the gradient size is about a 100th of optimizing pixel space.
            # 25 is too weak. get robot on either size.
            state.deepFeatures = state.deepFeatures - step_size * grad_cond * 200
        else:
            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
            print("gradient size: " + str(grad_cond.abs().mean().item()))
            print("gradient size: " + str(grad_cond.abs().sum().item()))
            latents = latents - step_size * grad_cond
        return latents

    def _perform_iterative_refinement_step(self,
                                           latents: torch.Tensor,
                                           loss: torch.Tensor,
                                           threshold: float,
                                           text_embeddings: torch.Tensor,
                                           text_input,
                                           attention_store: AttentionStore,
                                           step_size: float,
                                           t: int,
                                           attention_res: int = 16,
                                           smooth_attentions: bool = True,
                                           sigma: float = 0.5,
                                           kernel_size: int = 3,
                                           max_refinement_steps: int = 20,
                                           normalize_eot: bool = False):
        """
        Performs the iterative latent refinement introduced in the paper. Here, we continuously update the latent
        code according to our loss objective until the given threshold is reached for all tokens.
        """
        
        iteration = 0
        state.sub_iteration = iteration
        target_loss = max(0, 1. - threshold)
        while loss > target_loss:
            iteration += 1
            state.sub_iteration = iteration
            if state.optimizeDeepLatent:
                pass
                #latents = latents.clone().detach().requires_grad_(True)
            else:
                latents = latents.clone().detach().requires_grad_(True) #restart node graph
            if state.optimizeDeepLatent:
                state.deepFeatures = state.deepFeatures.clone().detach().requires_grad_(True)
                state.injectDeepFeatures = True

            #----optional, get without text condition to get diagnostic info from before optimizing...
            if state.config.diagnostic_level > 0:
                noise_pred_uncond = None
                with torch.no_grad(), state.TurnOffRequiresGradDeepLatent():
                    noise_pred_uncond = self.unet(latents, t,
                                                encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
                self.unet.zero_grad()
            #----

            # when optimizing we do not need the noise_pred (or the x0 pred)
            # we just need the attention maps to get our loss.
            
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            self.unet.zero_grad()
            
            if state.config.diagnostic_level > 0:
                with torch.no_grad():
                    noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
                    ddim_output = self.scheduler.step(noise_pred, t, latents)
                    self.save_image(ddim_output.pred_original_sample, "pred_pre_optim" + str(iteration))

            # Get max activation value for each subject token
            losses_dict = self._aggregate_and_get_max_attention_per_token(
                attention_store=attention_store,
                attention_res=attention_res,
                smooth_attentions=smooth_attentions,
                sigma=sigma,
                kernel_size=kernel_size,
                normalize_eot=normalize_eot
                )

            loss, losses = self._compute_loss(losses_dict, return_losses=True)

            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)

            if state.config.diagnostic_level > 0:
                with torch.no_grad(), state.TurnOffRequiresGradDeepLatent():
                    noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
                    noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

            try:
                low_token = np.argmax([l.item() if type(l) != int else l for l in losses])
            except Exception as e:
                print(e)  # catch edge case :)
                low_token = np.argmax(losses)

            #TODO multi token
            # low_word = self.tokenizer.decode(text_input.input_ids[0][indices_to_alter[low_token]])
            # print(f'\t Try {iteration}. {low_word} has a max attention of {losses_dict["max_loss"][low_token]}')
            # print(f'\t Try {iteration}. {low_word} has a has centroid of {losses_dict["col"][low_token]}')
            if iteration >= max_refinement_steps:
                print(f'\t Exceeded max number of iterations ({max_refinement_steps})! '
                      f'Finished with a max attention of {losses_dict["max_loss"][low_token]}')
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()

        # Get max activation value for each subject token
        max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
            attention_store=attention_store,
            attention_res=attention_res,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
        loss, losses = self._compute_loss(max_attention_per_index, return_losses=True)
        print(f"\t Finished with loss of: {loss}")
        state.sub_iteration = 0
        return loss, latents, max_attention_per_index

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[unet_2d_condition.UNet2DConditionOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        selfunet = self.unet
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**selfunet.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if selfunet.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = selfunet.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=selfunet.dtype)
        emb = selfunet.time_embedding(t_emb)

        if selfunet.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if selfunet.config.class_embed_type == "timestep":
                class_labels = selfunet.time_proj(class_labels)

            class_emb = selfunet.class_embedding(class_labels).to(dtype=selfunet.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = selfunet.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in selfunet.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = selfunet.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
        )

        prev = down_block_res_samples[-1]
        if state.optimizeDeepLatent:
            if state.injectDeepFeatures:
                list1 = list(down_block_res_samples)
                batch_size = down_block_res_samples[-1].shape[0]
                if batch_size > 1:
                    list1[-1] = state.deepFeatures.repeat(2,1,1,1) #.clone().detach().requires_grad_(True) #clone().detach() means that our original will not be part of the node graph.
                else:
                    list1[-1] = state.deepFeatures #.clone().detach().requires_grad_(True)
                down_block_res_samples = tuple(list1)
            else:
                state.deepFeatures = prev
            if state.deepLatentRequiresGrad:
                state.deepFeatures.requires_grad_(True)


        # 5. up
        for i, upsample_block in enumerate(selfunet.up_blocks):
            is_final_block = i == len(selfunet.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
        # 6. post-process
        sample = selfunet.conv_norm_out(sample)
        sample = selfunet.conv_act(sample)
        sample = selfunet.conv_out(sample)

        if not return_dict:
            return (sample,)

        return unet_2d_condition.UNet2DConditionOutput(sample=sample)


    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            attention_store: AttentionStore,
            attention_res: int = 16,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            max_iter_to_alter: Optional[int] = 25,
            run_standard_sd: bool = False,
            thresholds: Optional[dict] = {0: 0.05, 10: 0.5, 20: 0.8},
            scale_factor: int = 20,
            scale_range: Tuple[float, float] = (1., 0.5),
            smooth_attentions: bool = True,
            sigma: float = 0.5,
            kernel_size: int = 3,
            sd_2_1: bool = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        Examples:
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
            :type attention_store: object
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        self.unet.__dict__["forward"] = self.forward

        # 2. Define call parameters
        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_inputs, prompt_embeds = self._encode_prompt( #prompt embeds is negative (i.e.""), position
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
        # 4. Prepare timesteps #50 steps default
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                state.cur_time_step_iter = i
                with torch.enable_grad(): #this causes issues with save_image() since requires_grad is forced to true.
                    if state.optimizeDeepLatent:
                        latents = latents.detach()
                    else:
                        latents = latents.clone().detach().requires_grad_(True) #basically restart node graph
                    state.injectDeepFeatures = False #i.e. do not reuse the old noiser one.

                    if state.config.diagnostic_level > 0:
                        #----optional, get without text condition to get diagnostic info from before optimizing...
                        with torch.no_grad(), state.TurnOffRequiresGradDeepLatent():
                            noise_pred_uncond = self.unet(latents, t,
                                                        encoder_hidden_states=prompt_embeds[0].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                        self.unet.zero_grad()
                        #----

                    # Forward pass of denoising with text conditioning
                    noise_pred_text = self.unet(latents, t,
                                                encoder_hidden_states=prompt_embeds[1].unsqueeze(0), cross_attention_kwargs=cross_attention_kwargs).sample
                    #ddim_output = self.scheduler.step(noise_pred_text, t, latents, **extra_step_kwargs)
                    self.unet.zero_grad()

                    if state.config.diagnostic_level > 0:
                    #----diagnostics
                        with torch.no_grad():
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                            ddim_output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                            self.save_image(ddim_output.pred_original_sample, "pred_pre_optim")
                    #----


                    # Get max activation value for each subject token
                    max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                        attention_store=attention_store,
                        attention_res=attention_res,
                        smooth_attentions=smooth_attentions,
                        sigma=sigma,
                        kernel_size=kernel_size,
                        normalize_eot=sd_2_1)

                    if not run_standard_sd:

                        loss = self._compute_loss(losses_dict=max_attention_per_index)

                        # If this is an iterative refinement step, verify we have reached the desired threshold for all
                        if i in thresholds.keys() and loss > 1. - thresholds[i]:
                            del noise_pred_text
                            torch.cuda.empty_cache()
                            # here we return the new optimized latent
                            loss, latents, max_attention_per_index = self._perform_iterative_refinement_step(
                                latents=latents,
                                loss=loss,
                                threshold=thresholds[i],
                                text_embeddings=prompt_embeds,
                                text_input=text_inputs,
                                attention_store=attention_store,
                                step_size=scale_factor * np.sqrt(scale_range[i]),
                                t=t,
                                attention_res=attention_res,
                                smooth_attentions=smooth_attentions,
                                max_refinement_steps=20, #
                                sigma=sigma,
                                kernel_size=kernel_size,
                                normalize_eot=sd_2_1)

                        # Perform gradient update
                        if i < max_iter_to_alter:
                            loss = self._compute_loss(losses_dict=max_attention_per_index)
                            if loss != 0:
                                latents = self._update_latent(latents=latents, loss=loss,  #TODO: non iterative version fails
                                                              step_size=scale_factor * np.sqrt(scale_range[i]))
                            print(f'Iteration {i} | Loss: {loss.item():0.4f}') #.item() needed. tensor itself doesnt implement certain format operations

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual (with the optimized latent)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1 (with optimized latent AND noise pred from optimized latent)
                with torch.no_grad():
                    ddim_output = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                    latents = ddim_output.prev_sample
                    if state.config.diagnostic_level > 1:
                        self.save_image(latents, "xt") #looks just like pred but with added noise....
                    if state.config.diagnostic_level > 0:
                        self.save_image(ddim_output.pred_original_sample, "pred")

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        #image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        has_nsfw_concept = False
        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    



    def save_viridis(self, tensor1, tag):
        with torch.no_grad():
            x = tensor1 - tensor1.min()
            x = x / x.max()
            fname = tag + "_" + helpers.get_meta_prompt_clean() + state.get_name() + "_subiter_" + str(state.sub_iteration) + ".png"
            prompt_output_path = state.config.output_path / state.config.prompt / str(state.cur_seed)
            prompt_output_path.mkdir(exist_ok=True, parents=True)
            plt.imsave(prompt_output_path / fname, x.detach().cpu())

    def get_token(self, index):
        return self.tokenizer.decode(self.tokenizer(state.config.prompt)['input_ids'][index])

    def save_image(self, latent, tag):
        image = self.decode_latents(latent.detach())
        image = self.numpy_to_pil(image)
        #fname = state.config.prompt + "_iter" + str(state.cur_time_step_iter) + "_" + tag + "_seed" + str(state.cur_seed) + "_toRight" + str(state.toRight) + ".png"
        fname = helpers.get_meta_prompt_clean() + state.get_name() + "_" + tag
        fname = fname.replace('[','_').replace(']','_').replace(':','_').replace('.','_') + ".png"
        prompt_output_path = state.config.output_path / state.config.prompt / str(state.cur_seed)
        prompt_output_path.mkdir(exist_ok=True, parents=True)
        helpers.annotate_image(image)
        image[0].save(prompt_output_path / fname)
