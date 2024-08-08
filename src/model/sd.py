import torch
import transformers
transformers.logging.set_verbosity(transformers.logging.ERROR)
import torch.nn.functional as F

from torchvision import transforms
from diffusers import DDPMScheduler, AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPModel, CLIPFeatureExtractor, CLIPTokenizer
from model.sd_pipeline import pipe_train
from model.cliptextmodel import CLIPTextModel
from prompts.paths import MODEL_CACHE_DIR



def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)

    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def compute_spherical_dist_loss(image, prompt, clip_model, tokenizer, device):
    image_embeddings_clip = get_image_embeddings_clip(image, clip_model)
    text_embeddings_clip = get_text_embeddings_clip(prompt, tokenizer, clip_model, device)
    loss = spherical_dist_loss(image_embeddings_clip, text_embeddings_clip).mean()
    return loss


def get_image_embeddings_clip(image, clip_model):
    image_embeddings_clip = clip_model.get_image_features(image)
    image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(p=2, dim=-1, keepdim=True)
    return image_embeddings_clip


def get_text_embeddings_clip(prompt, tokenizer, clip_model, device):
    prompt_input_ids = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)
    text_embeddings_clip = clip_model.get_text_features(prompt_input_ids)
    text_embeddings_clip = text_embeddings_clip / text_embeddings_clip.norm(p=2, dim=-1, keepdim=True)
    return text_embeddings_clip


def inter_pipe_image_process(image, normalize, cut_out_size):
    from torchvision import transforms
    image = image.transpose(1, 3)
    image = transforms.Resize(cut_out_size)(image)
    image = normalize(image).to(image.dtype)
    return image


def get_stable_diffusion_pipeline(model_id, torch_dtype, cache_dir, device, return_modules=False):
    local_files_only = False
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", cache_dir=cache_dir)
    text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", revision=None, cache_dir=cache_dir, torch_dtype=torch_dtype, local_files_only=local_files_only,
        )
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", revision=None, cache_dir=cache_dir, torch_dtype=torch_dtype, local_files_only=local_files_only,
    )
    vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", revision=None, cache_dir=cache_dir, torch_dtype=torch_dtype, local_files_only=local_files_only,
        )
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        local_files_only=local_files_only,
    )
    pipe = pipe.to(device)
    ## lock model parameters to save space
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

    if return_modules:
        return tokenizer, text_encoder, unet, vae, pipe
    else:
        return pipe


class StableDiffusion():
    def __init__(self,
                 version,
                 torch_dtype=torch.float16,
                 device='cuda:0',
                 cache_dir=MODEL_CACHE_DIR):
        if version == 'v1-4' or version == 'v1':
            model_id = f"CompVis/stable-diffusion-v1-4"
        elif version == 'v2':
            model_id = f"stabilityai/stable-diffusion-2-base"
        elif version == 'xl':
            raise NotImplementedError('Still organizing, coming in the next patch')
        else:
            raise ValueError(version)

        ## args
        self.version = version
        self.device = device
        self.model_id = model_id

        ## sd pipeline
        tokenizer, text_encoder, unet, vae, pipe = get_stable_diffusion_pipeline(model_id, torch_dtype, cache_dir, device, True)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.pipe = pipe

        ## clip model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir).to(device)
        feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
        self.normalize = transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
        self.cut_out_size = (
            feature_extractor.size
            if isinstance(feature_extractor.size, int)
            else feature_extractor.size["shortest_edge"]
        )

    @torch.no_grad()
    def inference(self,
                  ori_prompt,
                  opt_prompt,
                  seed_list,
                  args,
                  negative_prompt=None,
                  prompt_embeds=None,
                  negative_prompt_embeds=None):
        task_name = args.task_config['task_name'].lower()
        image_pil_loss, avg_loss = [], 0
        for seed in seed_list:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            image = pipe_train(self.pipe, opt_prompt,
                            negative_prompt=negative_prompt,
                            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                            generator=generator)  ## tensor (1, 512, 512, 3)
            image_np = image.detach().cpu().numpy()
            image_pil = self.pipe.numpy_to_pil(image_np)[0]
        
            ## loss
            image = inter_pipe_image_process(image, self.normalize, self.cut_out_size)
            loss = compute_spherical_dist_loss(image, ori_prompt, self.clip_model, self.tokenizer, self.device)
            loss = -loss.item() if 'attack' in task_name else loss.item()

            image_pil_loss.append([image_pil, loss, seed])
            avg_loss += loss

        avg_loss /= len(seed_list)
        
        return avg_loss, image_pil_loss


    def inference_with_grad(self,
                            ori_prompt,
                            prompt_embeds,
                            negative_prompt_embeds,
                            seed_list,
                            t,
                            args):
        """ run diffusion model inference steps with backpropagation enabled """
        task_name = args.task_config['task_name'].lower()
        if not isinstance(prompt_embeds, list):
            prompt_embeds = [prompt_embeds] * len(seed_list)
            negative_prompt_embeds = [negative_prompt_embeds] * len(seed_list)
        assert len(prompt_embeds) == len(seed_list) and len(negative_prompt_embeds) == len(seed_list)

        avg_loss = 0
        for seed, pe, npe in zip(seed_list, prompt_embeds, negative_prompt_embeds):
            noise_scheduler = DDPMScheduler.from_pretrained(self.model_id, subfolder="scheduler", local_files_only=True)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            image = pipe_train(self.pipe,
                               prompt_embeds=pe,
                               negative_prompt_embeds=npe,
                               generator=generator,
                               noise_scheduler=noise_scheduler,
                               estimation_step=t)
            image = inter_pipe_image_process(image, self.normalize, self.cut_out_size)

            ### loss
            opt_image_embeddings_clip = get_image_embeddings_clip(image, self.clip_model)
            ori_text_embeddings_clip = get_text_embeddings_clip(ori_prompt, self.tokenizer, self.clip_model, self.device)

            loss = spherical_dist_loss(opt_image_embeddings_clip, ori_text_embeddings_clip).mean()
            loss = -1 * loss if 'attack' in task_name else loss

            total_loss = loss / len(seed_list)
            total_loss.backward(retain_graph=True)
            avg_loss += total_loss.item()
        
        return avg_loss
