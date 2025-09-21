import numpy.typing as npt
import torch
def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = torch.Tensor(dataset)
    # print(dataset)
    sts = torch.randint(
        low = 0,
        high = dataset.size(-1) - context_length,
        size = (batch_size,)
    )
    samples = torch.stack([dataset[st:st + context_length] for st in sts])
    labels = torch.stack([dataset[st + 1:st + context_length + 1] for st in sts])
    return (samples,labels)