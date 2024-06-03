# Author: Nathan Trouvain at 10/10/23 <nathan.trouvain<at>inria.fr>
# Licence: MIT License
# Copyright: Nathan Trouvain
# Author: Nathan Trouvain at 29/08/2023 <nathan.trouvain<at>inria.fr>
import lightning as pl
import torch
import torchmetrics

from torch.nn import functional as F
from torch import nn
from torchaudio import transforms

from ..gan.params import SliceLengths


class SyllableClassifier(nn.Module):
    """
    Simple convnet for canary song syllables classification. Used to compute
    Inception Score over GAN generated sounds.

    Replication of Donahue et al. (2018) method.

    Parameters
    ----------
    n_classes : int, default to 16
        Number of distinct syllable classes in the training dataset.
    sample_rate : int, default to 16000
        Audio sampling rate, in Hz.
    n_fft : int, default to 1024
        Number of FFT bins to compute when preprocessing audio.
    win_length : int, default to 1024
        FFT window length.
    hop_length : int, default to 128
        FFT stride size.
    f_min : int, default to 40
        Frequency lower bound, in Hz.
    f_max : int, default to 7800
        Frequency upper bound, in Hz.
    n_mels : int, default to 128
        Number of Mel-frequency bins.
    pad_mode : str, default to "constant"
        Padding mode.
    center : bool, default to True
        If True, then each spectrogram step is centered
        in the audio window.
    slice_length : SliceLengths
        Audio sample length, in number of timesteps.
    n_channels : int, default to 128
        Depth of convolution operation.
    """

    def __init__(
        self,
        n_classes=16,
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=128,
        f_min=40,
        f_max=7800,
        n_mels=128,
        pad_mode="constant",
        center=True,
        slice_length=SliceLengths.SHORT,
        n_channels=128,
    ):
        super(SyllableClassifier, self).__init__()

        # [1, slice_length] -> [1, n_mels=128, 129=513[n_fft // 2 +1]//2+1]
        self.melspec = transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            pad_mode=pad_mode,
            center=center,
        )

        self.convnet = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, n_channels, (5, 5), padding="same"),
            nn.ReLU(),
            # [1, 128, 129] -> [128, 64, 64]
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels, (5, 5), padding="same"),
            nn.ReLU(),
            # [128, 64, 64] -> [128, 32, 32]
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels, (5, 5), padding="same"),
            nn.ReLU(),
            # [128, 32, 32] -> [128, 16, 16]
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.BatchNorm2d(n_channels),
            nn.Conv2d(n_channels, n_channels, (5, 5), padding="same"),
            nn.ReLU(),
            # [128, 16, 16] -> [128, 8, 8]
            nn.MaxPool2d((2, 2), (2, 2)),
        )

        # width reduction factor = (length // hop_length) // 2**(n_layers)
        w_fact = (slice_length.value // hop_length) // 2**4
        # heigth reduction factor = n_mels // 2**(n_layers)
        h_fact = n_mels // 2**4

        self.logits = nn.Sequential(
            nn.Flatten(),  # [128, 8, 8] -> [128 * 8 * 8]
            nn.BatchNorm1d(n_channels * w_fact * h_fact),
            nn.Linear(n_channels * w_fact * h_fact, n_classes),
        )

    def _init_conv_layers(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        mel = self.melspec(x)
        h = self.convnet(mel)
        return self.logits(h)


class LightningSyllableClassifier(pl.LightningModule):
    """
    Lightning Module wrapper around SyllableClassifier.

    Parameters
    ----------
    n_classes : int, default to 16
        Number of distinct syllable classes in the training dataset.
    sample_rate : int, default to 16000
        Audio sampling rate, in Hz.
    n_fft : int, default to 1024
        Number of FFT bins to compute when preprocessing audio.
    win_length : int, default to 1024
        FFT window length.
    hop_length : int, default to 128
        FFT stride size.
    f_min : int, default to 40
        Frequency lower bound, in Hz.
    f_max : int, default to 7800
        Frequency upper bound, in Hz.
    n_mels : int, default to 128
        Number of Mel-frequency bins.
    pad_mode : str, default to "constant"
        Padding mode.
    center : bool, default to True
        If True, then each spectrogram step is centered
        in the audio window.
    slice_length : SliceLengths
        Audio sample length, in number of timesteps.
    n_channels : int, default to 128
        Depth of convolution operation.
    lr : float, default to 1e-4
        Learning rate.
    seed : int, default to 0
        Random state seed.
    """

    def __init__(
        self,
        n_classes=16,
        sample_rate=16000,
        n_fft=1024,
        win_length=1024,
        hop_length=128,
        f_min=40,
        f_max=7800,
        n_mels=128,
        pad_mode="constant",
        center=True,
        lr=1e-4,
        seed=0,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.classifier = SyllableClassifier(
            n_classes=n_classes,
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            pad_mode=pad_mode,
            center=center,
        )

        # Logging metrics during training
        self.train_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=n_classes
        )
        self.val_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=n_classes
        )

        self.train_f1_score = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=n_classes,
        )
        self.val_f1_score = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=n_classes,
        )

        self.test_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=n_classes
        )
        self.test_f1_score = torchmetrics.classification.F1Score(
            task="multiclass",
            num_classes=n_classes,
        )

    def forward(self, x):
        y_pred = self.classifier(x)
        return torch.argmax(F.softmax(y_pred, dim=1), dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.classifier(x)
        loss = F.cross_entropy(logits, y)
        y_pred = torch.argmax(logits, dim=1)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        self.train_accuracy(y_pred, y)
        self.train_f1_score(y_pred, y)

        # No need to sync_dist, torchmetrics handles the sync for us
        self.log(
            "train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("train_f1", self.train_f1_score, on_step=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x_val, y_val = batch
        logits = self.classifier(x_val)
        loss = F.cross_entropy(logits, y_val)
        y_pred = torch.argmax(logits, dim=1)

        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        self.val_accuracy(y_pred, y_val)
        self.val_f1_score(y_pred, y_val)

        self.log(
            "val_acc",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("val_f1", self.val_f1_score, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x_test, y_test = batch
        logits = self.classifier(x_test)
        loss = F.cross_entropy(logits, y_test)
        y_pred = torch.argmax(logits, dim=1)

        self.log("test_loss", loss, sync_dist=True)

        self.test_accuracy(y_pred, y_test)
        self.test_f1_score(y_pred, y_test)

        self.log("test_acc", self.test_accuracy)
        self.log("test_f1", self.test_f1_score)

    def predict_step(self, batch, batch_idx):
        return self.classifier(batch)

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt = torch.optim.SGD(self.classifier.parameters(), lr=lr)
        return opt
