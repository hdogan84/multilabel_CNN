import torch
from torch import tensor
from typing import Tuple


def pool_by_segments(
    samples: tensor, grouping_indices: tensor, pooling_method="mean"
) -> tuple:
    """ select mean(samples) | meanexp(samples) | max(samples)  , count() from samples group by grouping_indices order by grouping_indices asc """
    input_type = None
    if samples.dtype != torch.float32:
        input_type = samples.dtype
        samples = samples.type(torch.float32)
    # samples have to be two dimensonal beacuse auf matrix multiplication
    samples_unsqueezed = False
    if samples.dim() == 1:
        samples_unsqueezed = True
        samples = samples.unsqueeze(1)

    weight = torch.zeros(
        grouping_indices.max() + 1, samples.shape[0], device=samples.device
    )
    # Creates for every label a weight vector with one if sample is from it
    weight[grouping_indices, torch.arange(samples.shape[0], device=samples.device)] = 1

    # count how often a label occures
    label_count = weight.sum(dim=1)

    if pooling_method == "max":
        result_values = torch.stack(
            [torch.max(samples[indices > 0], 0) for indices in weight]
        )
    elif pooling_method == "mean":
        result_values = torch.stack(
            [torch.mean(samples[indices > 0], 0) for indices in weight]
        )
    elif pooling_method == "meanexp":  # cases mean and meanexp
        result_values = torch.stack(
            [
                torch.mean(
                    torch.square(samples[indices > 0])
                    * torch.sign(samples[indices > 0]),
                    0,
                )
                for indices in weight
            ]
        )
    else:
        raise ValueError("pooling_method")

    # create index of
    index = torch.arange(result_values.shape[0], device=samples.device)[label_count > 0]
    if samples_unsqueezed:
        result_values = result_values.squeeze(1)
    if input_type is not None:
        result_values = result_values.type(input_type)
    return result_values[index], label_count[index]


def to_one_hot_encoding(class_indices: tensor, num_classes: int) -> tensor:
    class_matrix = torch.zeros(
        class_indices.shape[0], num_classes, device=class_indices.device
    )
    class_matrix[range(class_indices.shape[0]), class_indices] = 1
    return class_matrix
