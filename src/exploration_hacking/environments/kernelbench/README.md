## KernelBench

The scaffold was adopted from the [Environment Hub](https://app.primeintellect.ai/dashboard/environments/primeintellect/kernelbench). I've made many changes so that it is compatible with the backend used in this repo. Currently it uses `SingleTurnEnv` and the reward from Kevin-32B paper (i.e. 0.3 for correct solution + runtime improvement). See this for more [info](https://arxiv.org/abs/2507.11948).


### Evaluation
The CUDA kernel evals are done through RunPod Serverless API. The Docker image can be found [here](https://hub.docker.com/repository/docker/yoenoo/runpod-serverless-test/general) (use the latest version).

Some pointers:
- Active endpoint: https://console.runpod.io/serverless/user/endpoint/7slsfq3i9eqy0x?tab=overview
- Dockerfile: https://github.com/yoenoo/exploration_hacking/blob/verifiers/Dockerfile
- Docker image build script: https://github.com/yoenoo/exploration_hacking/blob/verifiers/push.sh
- [rp_handler.py](https://github.com/yoenoo/exploration_hacking/blob/verifiers/rp_handler.py): this is handling the evals
  - see [this](https://docs.runpod.io/serverless/workers/handler-functions) for more info on how to build handlers