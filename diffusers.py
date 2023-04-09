#import time
#import intel_extension_for_pytorch as ipex
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

"""
def elapsed_time(pipeline, prompt, nb_pass=10, num_inference_steps=20):
	# warmup
	images = pipeline(prompt, num_inference_steps=10).images
	start = time.time()
	for _ in range(nb_pass):
		_ = pipeline(prompt, num_inference_steps=num_inference_steps, output_type="np")
	end = time.time()
	return (end - start) / nb_pass
"""

use_cuda = True
model_id = "stabilityai/stable-diffusion-2-1"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

"""
if use_cuda:
	pipe = pipe.to("cuda")
else:
	# to channels last
	pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
	pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
	pipe.text_encoder = pipe.text_encoder.to(memory_format=torch.channels_last)
	pipe.safety_checker = pipe.safety_checker.to(memory_format=torch.channels_last)

	# Create random input to enable JIT compilation
	sample = torch.randn(2,4,64,64)
	timestep = torch.rand(1)*999
	encoder_hidden_status = torch.randn(2,77,768)
	input_example = (sample, timestep, encoder_hidden_status)

	# optimize with IPEX
	pipe.unet = ipex.optimize(pipe.unet.eval(), dtype=torch.bfloat16, inplace=True, sample_input=input_example)
	pipe.vae = ipex.optimize(pipe.vae.eval(), dtype=torch.bfloat16, inplace=True)
	pipe.text_encoder = ipex.optimize(pipe.text_encoder.eval(), dtype=torch.bfloat16, inplace=True)
	pipe.safety_checker = ipex.optimize(pipe.safety_checker.eval(), dtype=torch.bfloat16, inplace=True)
"""

pipe.enable_xformers_memory_efficient_attention() # use less memory to build larger image
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config) # use best scheduler recommended by huggingface

print(elapsed_time(pipe, "sailing ship in storm by Rembrandt"))
"""
prompt = "one astronaut in the moon, (wide shot), full body, two legs, realistic, best quality, (intricate details:1.3), hyper detail, finely detailed, colorful, (studio lighting:1.2), (Fujifilm XT3), (photorealistic:1.3), (detailed skin:1.2)"
image = pipe(prompt, width=512, height=512).images[0]

image.save("result.png")
"""