# External components

FutureGain uses separately installed third-party software and externally supplied
model weights and datasets. No upstream source, model weight or dataset is bundled.

- FAR: https://github.com/showlab/FAR at e59536ac30e5e3fa93db756fccfd422420a65ff6.
- FAR model assets: https://huggingface.co/guyuchao/FAR_Models.
- BAIR data mirror: https://huggingface.co/datasets/guyuchao/BAIR.
- The auxiliary MCVD adapter expects https://github.com/voletiv/mcvd-pytorch
  at 451da2eb635bad50da6a7c03b443a34c6eb08b3a.
- Metric implementations use LPIPS, TorchMetrics and externally supplied I3D weights.

Follow each provider's license and data-use terms. This source distribution does
not grant rights to external assets. No new software license is declared for the
original FutureGain code in this initial repository.
