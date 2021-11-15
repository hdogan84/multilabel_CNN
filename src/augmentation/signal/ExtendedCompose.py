import inspect
import random
from inspect import signature
from warnings import catch_warnings
import warnings


class ExtendedCompose:
    def __init__(self, transforms, p=1.0, shuffle=False):

        self.p = p
        self.shuffle = shuffle

        name_list = []
        # Anaylse callable agurment lent and save it
        transform_args_len_list = []
        randomize_args_len_list = []
        for transform in transforms:
            argspec = inspect.getfullargspec(transform)
            if "self" in argspec.args:
                argspec.args.remove("self")
            transform_args_len_list.append(len(argspec.args))

            argspec = inspect.getfullargspec(transform)
            if "self" in argspec.args:
                argspec.args.remove("self")
            randomize_args_len_list.append(len(argspec.args))

            name_list.append(type(transform).__name__)

        self.__name__ = "_".join(name_list)
        self.transforms = list(
            zip(transforms, transform_args_len_list, randomize_args_len_list)
        )

    def __call__(self, samples, sample_rate, y=None):
        transforms = self.transforms.copy()

        if random.random() < self.p:

            if self.shuffle:
                random.shuffle(transforms)
            for transform, args_len, _ in transforms:
                try:
                    if args_len == 2:
                        samples = transform(samples, sample_rate)
                        return samples, y
                    else:
                        samples, y = transform(samples, sample_rate, y)

                        return samples, y  # samples, y_n
                except Exception as e:
                    if str(e) == "local variable 'data' referenced before assignment":
                        # catch error in scipy wav read function reason is probaly broken wav file
                        pass
                    else:
                        raise e

        return samples, y

    def randomize_parameters(self, samples, sample_rate, y=None):
        """
        Randomize and define parameters of every transform in composition.
        """
        for transform, _, args_len in self.transforms:
            if args_len == 2:
                transform.randomize_parameters(samples, sample_rate)
            else:
                transform.randomize_parameters(samples, sample_rate, y)

    def freeze_parameters(self):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect chain with the exact same parameters to multiple
        sounds.
        """
        for transform, _ in self.transforms:
            transform.freeze_parameters()

    def unfreeze_parameters(self):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        for transform, _ in self.transforms:
            transform.unfreeze_parameters()
