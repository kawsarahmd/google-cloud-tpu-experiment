# Google Cloud TPU Experiment Repository

Welcome to the Google Cloud TPU Experiment Repository. This repository is dedicated to experiments using JAX/Flax on Google Cloud TPUs, which leverage the exceptional processing power of TPUs to accelerate deep learning tasks.

## Overview

This repository contains experiments conducted with the JAX/Flax library, chosen for its superior performance in a TPU environment. The experiments focus on demonstrating the speed and efficiency of JAX/Flax on TPUs compared to other environments.


## TPU Restart Command Example
To restart your TPU VM, use the following bash command:
```bash
gcloud compute tpus tpu-vm stop tpu-vm-name --zone=us-central2-b && \
gcloud compute tpus tpu-vm start tpu-vm-name --zone=us-central2-b




