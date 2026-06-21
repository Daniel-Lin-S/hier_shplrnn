from torch import nn
from abc import ABC, abstractmethod


class BaseFoundationModel(nn.Module, ABC):
    """
    Abstract base class for foundation models.
    """

    @abstractmethod
    def load_pretrained_weights(
        self,
        pretrained_weights_path: str,
        device: str = 'cpu',
        verbose: bool = True,
        proj: bool = False,
        **kwargs
    ) -> None:
        """
        Load pretrained weights into the model.

        Parameters
        ----------
        pretrained_weights_path : str, optional
            Path to the pretrained weights file.
            If not provided, no weights will be loaded.
        device : str, optional
            Device to load the weights onto. Default is 'cpu'.
        verbose : bool, optional
            Whether to print loading information. Default is True.
        proj: bool=False
            If True, use the model for projection to out_dim,
            used for reconstruction.
            If False, modify the model to output features
            with dimension d_model for feature extraction.
            Default is False.
        **kwargs
            Additional keyword arguments specific to the model.
        """
        pass
