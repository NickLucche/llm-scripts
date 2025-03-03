# Run on CUDA and TPU separately to compare performance.
import torch
from vllm.platforms import current_platform
if current_platform.is_tpu():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

from vllm.v1.sample.sampler import Sampler, SamplingMetadata
from time import time

# Show off time difference in Sampler code when topp/topk is enabled

DEVICE = xm.xla_device() if current_platform.is_tpu() else torch.device("cuda:0")
# Compile ""model"" and sampler init
sampler = Sampler().to(DEVICE)
V = 128256
H = V//32
model = torch.nn.Linear(H, V, device=DEVICE)
def debug_barrier(start=None, name=""):
    if current_platform.is_tpu():
        xm.mark_step()
        xm.wait_device_ops()
    if start:
        print(name, "elapsed time:", time()-start)
    # if current_platform.is_tpu():
    #     print(met.short_metrics_report())
    #     met.clear_all()
debug_barrier(name='Model+Sampler')

@torch.no_grad()
def model_with_sample(model: torch.nn.Module, x: torch.Tensor, meta: SamplingMetadata):
    y = model(x)
    y = sampler(y, meta)
    return y

# compile the model
if current_platform.is_tpu():
    model = torch.compile(model, backend="openxla",
                                    fullgraph=True,
                                    dynamic=False)
else:
    model = torch.compile(model)

temp = torch.tensor([0.4]).to(DEVICE)
topp = torch.tensor([0.6]).to(DEVICE) # some value just for tracing
topk = torch.tensor([12], dtype=torch.long).to(DEVICE) # some value just for tracing
meta = SamplingMetadata(temp, False, False, spec_token_ids=None, top_p=topp, top_k=topk, min_p=None, generators={}, max_num_logprobs=None, 
                        no_penalties=True, frequency_penalties=None, presence_penalties=None, repetition_penalties=None, output_token_ids=[[]], min_tokens=None, 
                        logit_bias=[], allowed_token_ids_mask=None, prompt_token_ids=None)
debug_barrier(name="Meta") # Nothing is executed nor compiled(!), things are just moved to device

## Compile graph
for B in [1, 4, 16, 32]: 
    x = torch.randn(B, H, device=DEVICE)
    debug_barrier() # tensor already on device graph
    s = time()
    out = model_with_sample(model, x, meta)
    debug_barrier(s, f"Compiling/Warmup {B}")

# Run
times = dict()
for B in [1, 4, 16, 32]:
    x = torch.randn(B, H, device=DEVICE)
    debug_barrier() # tensor already on device graph
    for _ in range(4):
        s = time()
        out = model_with_sample(model, x, meta)
        debug_barrier(s, f"Running {B}")
