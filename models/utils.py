from typing import List
import torch.nn as nn


def load_state_dict(
        model: nn.Module, state_dict: dict,
        prefix: str='',
        ignore_missing: List[str]=[],
        verbose: bool=True
    ) -> None:
    """
    Load a state_dict into a model in place, handling missing and unexpected keys.

    Parameters
    ----------
    model : torch.nn.Module
        The model into which the state_dict will be loaded.
    state_dict : dict
        The state_dict containing the weights to load.
    prefix : str, optional
        A prefix to prepend to all keys in the state_dict, by default ''.
        Useful when loading weights into submodules.
    ignore_missing : List[str], optional
        A list of substrings to ignore when checking for missing keys,
        by default [].
    verbose : bool, optional
        Whether to print detailed error messages, by default True.
    """
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()

    def load_with_prefix(module: nn.Module, prefix: str=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load_with_prefix(child, prefix + name + '.')

    load_with_prefix(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing:
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if verbose:
        if len(missing_keys) > 0:
            print("Weights of {} not initialised from pretrained model: {}".format(
                model.__class__.__name__, missing_keys), flush=True)
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys), flush=True)
        if len(ignore_missing_keys) > 0:
            print("Ignored weights of {} not initialised from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys), flush=True)
        if len(error_msgs) > 0:
            print('\n'.join(error_msgs), flush=True)
