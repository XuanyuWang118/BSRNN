import logging
from itertools import accumulate

import librosa
import numpy as np
import torch
import torch.nn as nn

from tools.tcn import choose_norm as choose_norm1d

EPS = torch.finfo(torch.get_default_dtype()).eps


class BSRNN(nn.Module):
    def __init__(
        self,
        input_dim=481,
        num_channel=16,
        num_layer=6,
        target_fs=48000,
        subbands=None,
        causal=False,
        num_spk=1,
        norm_type="GN",
    ):
        """Band-Split RNN (BSRNN).

        References:
            [1] J. Yu, H. Chen, Y. Luo, R. Gu, and C. Weng, “High fidelity speech
            enhancement with band-split RNN,” in Proc. ISCA Interspeech, 2023.
            https://isca-speech.org/archive/interspeech_2023/yu23b_interspeech.html
            [2] J. Yu, and Y. Luo, “Efficient monaural speech enhancement with
            universal sample rate band-split RNN,” in Proc. ICASSP, 2023.
            https://ieeexplore.ieee.org/document/10096020

        Args:
            input_dim (int): maximum number of frequency bins corresponding to
                `target_fs`
            num_channel (int): embedding dimension of each time-frequency bin
            num_layer (int): number of time and frequency RNN layers
            target_fs (int): maximum sampling frequency supported by the model
            subbands (list or tuple, optional): list of subband sizes to split the
                frequency band into. If specified, this will override the subband
                definition in the `BandSplit` class.
            causal (bool): Whether or not to adopt causal processing
                if True, LSTM will be used instead of BLSTM for time modeling
            num_spk (int): number of outputs to be generated
            norm_type (str): type of normalization layer (cfLN / cLN / BN / GN)
        """
        super().__init__()
        norm1d_type = norm_type if norm_type != "cfLN" else "cLN"
        if causal and norm_type not in ("cfLN", "cLN"):
            raise ValueError(
                "For causal processing, only 'cfLN' and 'cLN' should be used"
                "to ensure causality."
            )
        self.num_layer = num_layer
        assert subbands is None or isinstance(subbands, (list, tuple)), subbands
        self.band_split = BandSplit(
            input_dim,
            target_fs=target_fs,
            subbands=subbands,
            channels=num_channel,
            norm_type=norm1d_type,
        )
        self.target_fs = target_fs
        self.causal = causal
        self.num_spk = num_spk

        self.norm_time = nn.ModuleList()
        self.rnn_time = nn.ModuleList()
        self.fc_time = nn.ModuleList()
        self.norm_freq = nn.ModuleList()
        self.rnn_freq = nn.ModuleList()
        self.fc_freq = nn.ModuleList()
        hdim = 2 * num_channel
        for i in range(self.num_layer):
            self.norm_time.append(choose_norm(norm_type, num_channel))
            self.rnn_time.append(
                nn.LSTM(
                    num_channel,
                    hdim,
                    batch_first=True,
                    bidirectional=not causal,
                )
            )
            self.fc_time.append(nn.Linear(hdim if causal else hdim * 2, num_channel))
            self.norm_freq.append(choose_norm(norm_type, num_channel))
            self.rnn_freq.append(
                nn.LSTM(num_channel, hdim, batch_first=True, bidirectional=True)
            )
            self.fc_freq.append(nn.Linear(4 * num_channel, num_channel))

        self.mask_decoder = MaskDecoder(
            input_dim,
            self.band_split.subbands,
            channels=num_channel,
            num_spk=num_spk,
            norm_type=norm1d_type,
        )

    def forward(self, x, fs=None):
        """BSRNN forward.

        Args:
            x (torch.Tensor): input tensor of shape (B, T, F, 2)
            fs (int, optional): sampling rate of the input signal.
                if not None, the input signal will be truncated to only process the
                effective frequency subbands.
                if None, the input signal is assumed to be already truncated to only
                contain effective frequency subbands.
        Returns:
            out (torch.Tensor): output tensor of shape (B, num_spk, T, F, 2)
        """
        # --- [TODO: 阶段 1 - 子带特征映射] ---
        # 1. 调用 self.band_split 获得隐含特征 z
        # 提示：z 的维度应为 (B, N, T, K)，其中 N 是通道数，K 是子带数
        # --- [TODO START] ---

        # --- [TODO END] ---
        
        B, N, T, K = z.shape
        skip = z

        # --- [TODO: 阶段 2 - 双路径循环建模] ---
        # 2. 遍历 self.num_layer 层，交替进行时间轴和频率轴建模
        # 提示：
        #    - 时域建模时，将维度变换为 (B*K, T, N)，使用 self.rnn_time[i]
        #    - 频域建模时，将维度变换为 (B*T, K, N)，使用 self.rnn_freq[i]
        #    - 每一部分建模后都要经过对应的 FC 层并进行残差连接 (skip = skip + out)
        # --- [TODO START] ---

        # --- [TODO END] ---

        # --- [TODO: 阶段 3 - Mask 预测与信号重构] ---
        # 3. 调用 self.mask_decoder 获得 mask (m) 和 residual (r)
        # 4. 将 m, r 和原始输入 x 转换为复数域进行计算 (out = m * x + r)
        # 5. 最后使用 torch.view_as_real 转回实数表示
        # --- [TODO START] ---

        # --- [TODO END] ---
        return ret


class BandSplit(nn.Module):
    def __init__(
        self, input_dim, target_fs=48000, subbands=None, channels=128, norm_type="GN"
    ):
        super().__init__()
        assert input_dim % 2 == 1, input_dim
        n_fft = (input_dim - 1) * 2
        # freq resolution = target_fs / n_fft = freqs[1] - freqs[0]
        freqs = torch.fft.rfftfreq(n_fft, 1.0 / target_fs).tolist()
        if subbands is not None:
            """This can be used to define custom subbands.
            For example, you can specify mel-style and ERB-style subbands via:

            subbands = get_mel_subbands(input_dim, n_mels=40, target_fs=48000)
            subbands = get_erb_subbands(
                input_dim, min_freq_idx=41, n_erbs=64, target_fs=48000
            )
            """
            assert isinstance(subbands, (list, tuple)) and len(subbands) > 0, subbands
            self.subbands = tuple(tuple(tup) for tup in subbands)
        elif input_dim == 481 and target_fs == 48000:
            """n_fft=960 (20ms) for 48000 Hz. The non-overlapping subbands are:

            first 20 200Hz subbands: [0-200], (200-400], (400-600], ..., (3800-4000]
            subsequent 6 500Hz subbands: (4000, 4500], ..., (6500, 7000]
            subsequent 7 2kHz subbands: (7000, 9000], ..., (19000, 21000]
            final 3kHz subband: (21000, 24000]
            """
            # size of each non-overlapping subband
            self.subbands = tuple([5] + [4] * 19 + [10] * 6 + [40] * 7 + [60])
            # convert subband sizes to the corresponding start and end indices
            self.subbands = self.convert_size_to_indices(self.subbands, input_dim)
        else:
            raise NotImplementedError(
                f"Please define your own subbands for input_dim={input_dim} and "
                f"target_fs={target_fs}"
            )

        self.validate_subbands(self.subbands, input_dim, target_fs=target_fs)
        # [start, end] frequencies of the subbands
        self.subband_freqs = tuple((freqs[st], freqs[et]) for st, et in self.subbands)

        self.norm = nn.ModuleList()
        self.fc = nn.ModuleList()
        for st, et in self.subbands:
            subband = et - st + 1
            self.norm.append(choose_norm1d(norm_type, int(subband * 2)))
            self.fc.append(nn.Conv1d(int(subband * 2), channels, 1))

    @staticmethod
    def convert_size_to_indices(subbands, input_dim):
        """Convert subband sizes to the corresponding start and end indices."""
        assert sum(subbands) == input_dim, (sum(subbands), input_dim)
        cumsum = tuple(accumulate((0,) + subbands))
        subbands = tuple(
            [(cumsum[i], cumsum[i + 1] - 1) for i in range(len(cumsum) - 1)]
        )
        return subbands

    @staticmethod
    def validate_subbands(subbands, input_dim, target_fs=48000):
        """Validate the subbands."""
        assert input_dim % 2 == 1, input_dim
        n_fft = (input_dim - 1) * 2
        freqs = torch.fft.rfftfreq(n_fft, 1.0 / target_fs).tolist()

        # 1. subband is a tuple of (start, end) indices
        assert all(len(subband) == 2 for subband in subbands), subbands
        # 2. the entire frequency range should be covered
        if subbands[-1][1] != input_dim - 1:
            raise ValueError(
                f"The last subband should cover the full frequency range, "
                f"but got {subbands[-1][1]} != {input_dim}"
            )

        st, et = 0, -1
        msg = ""
        has_overlap = False
        for i, subband in enumerate(subbands):
            # 3. start and end indices are inclusive
            # 4. all subbands should be sorted in ascending order
            assert subband[0] >= st, (i, subbands, st)
            assert subband[1] > et, (i, subbands, et)
            if i > 0:
                if subband[0] <= et:
                    has_overlap = True
                elif subband[0] > et + 1:
                    raise ValueError(
                        f"Subband {i} starts at {subband[0]} which is not "
                        f"adjacent to the previous subband ending at {et}"
                    )
            st, et = subband
            hz1, hz2 = freqs[st], freqs[et]
            msg += f"\tSubband {i}: {subband[0]} ~ {subband[1]} ({hz1} Hz ~ {hz2} Hz)\n"

        adj = "overlapping" if has_overlap else "non-overlapping"
        logging.info(f"Using {adj} subbands (num={len(subbands)}):\n{msg}")

    def forward(self, x, fs=None):
        """BandSplit forward.

        Args:
            x (torch.Tensor): input tensor of shape (B, T, F, 2)
            fs (int, optional): sampling rate of the input signal.
                if not None, the input signal will be truncated to only process the
                effective frequency subbands.
                if None, the input signal is assumed to be already truncated to only
                contain effective frequency subbands.
        Returns:
            z (torch.Tensor): output tensor of shape (B, N, T, K)
        """
        # --- [TODO: 实现子带切分与特征投影] ---
        # 1. 遍历 self.subbands，根据 [st, et] 索引截取对应的频率切片
        # 2. 将实部和虚部展平/拼接到通道维
        # 3. 应用 self.norm[i] 和 self.fc[i] (1D 卷积要求输入形状为 B, C, L)
        # 4. 处理 fs 引起的频率截断（可选/进阶）
        # 5. 将所有子带特征拼接在最后一个维度 K 上
        # --- [TODO START] ---

        # --- [TODO END] ---
        return z


class MaskDecoder(nn.Module):
    def __init__(self, freq_dim, subbands, channels=128, num_spk=1, norm_type="GN"):
        super().__init__()
        # The last subband should cover the entire frequency range
        assert freq_dim == subbands[-1][1] + 1, (freq_dim, subbands)
        self.subbands = subbands
        self.freq_dim = freq_dim
        self.num_spk = num_spk
        self.mlp_mask = nn.ModuleList()
        self.mlp_residual = nn.ModuleList()
        for st, et in self.subbands:
            subband = et - st + 1
            self.mlp_mask.append(
                nn.Sequential(
                    choose_norm1d(norm_type, channels),
                    nn.Conv1d(channels, 4 * channels, 1),
                    nn.Tanh(),
                    nn.Conv1d(4 * channels, int(subband * 4 * num_spk), 1),
                    nn.GLU(dim=1),
                )
            )
            self.mlp_residual.append(
                nn.Sequential(
                    choose_norm1d(norm_type, channels),
                    nn.Conv1d(channels, 4 * channels, 1),
                    nn.Tanh(),
                    nn.Conv1d(4 * channels, int(subband * 4 * num_spk), 1),
                    nn.GLU(dim=1),
                )
            )

    def forward(self, x):
        """MaskDecoder forward.

        Args:
            x (torch.Tensor): input tensor of shape (B, N, T, K)
        Returns:
            m (torch.Tensor): output mask of shape (B, num_spk, T, F, 2)
            r (torch.Tensor): output residual of shape (B, num_spk, T, F, 2)
        """
        # --- [TODO: 将隐含特征还原为全频段 Mask 和 Residual] ---
        # 1. 初始化全零的 m 和 r 容器，形状对应 (B, T, num_spk, F, 2)
        # 2. 遍历子带，取出 x 对应的分片 x[:, :, :, i]
        # 3. 通过 mlp_mask[i] 和 mlp_residual[i] 获取该频段的输出
        # 4. Reshape 并填入 m 和 r 的对应频率位置
        # --- [TODO START] ---

        # --- [TODO END] ---
        return m.moveaxis(1, 2), r.moveaxis(1, 2)


def choose_norm(norm_type, channel_size, shape="BDTF"):
    """The input of normalization will be (M, C, K), where M is batch size.

    C is channel size and K is sequence length.
    """
    if norm_type == "cfLN":
        return ChannelFreqwiseLayerNorm(channel_size, shape=shape)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size, shape=shape)
    elif norm_type == "BN":
        # Given input (M, C, T, K), nn.BatchNorm2d(C) will accumulate statics
        # along M, T, and K, so this BN usage is right.
        return nn.BatchNorm2d(channel_size)
    elif norm_type == "GN":
        return nn.GroupNorm(1, channel_size)
    else:
        raise ValueError("Unsupported normalization type")


class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)."""

    def __init__(self, channel_size, shape="BDTF"):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1, 1))  # [1, N, 1]
        self.reset_parameters()
        assert shape in ["BDTF", "BTFD"], shape
        self.shape = shape

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, y):
        """Forward.

        Args:
            y: [M, N, T, K], M is batch size, N is channel size, T and K are lengths

        Returns:
            cLN_y: [M, N, T, K]
        """

        assert y.dim() == 4

        if self.shape == "BTFD":
            y = y.moveaxis(-1, 1)

        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, T, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, T, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        if self.shape == "BTFD":
            cLN_y = cLN_y.moveaxis(1, -1)

        return cLN_y


class ChannelFreqwiseLayerNorm(nn.Module):
    """Channel-and-Frequency-wise Layer Normalization (cfLN)."""

    def __init__(self, channel_size, shape="BDTF"):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1, 1))  # [1, N, 1, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1, 1))  # [1, N, 1, 1]
        self.reset_parameters()
        assert shape in ["BDTF", "BTFD"], shape
        self.shape = shape

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, y):
        """Forward.

        Args:
            y: [M, N, T, K], M is batch size, N is channel size, T and K are lengths

        Returns:
            gLN_y: [M, N, T, K]
        """
        if self.shape == "BTFD":
            y = y.moveaxis(-1, 1)

        mean = y.mean(dim=(1, 3), keepdim=True)  # [M, 1, T, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=(1, 3), keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        if self.shape == "BTFD":
            gLN_y = gLN_y.moveaxis(1, -1)
        return gLN_y


def get_mel_subbands(input_dim, n_mels=40, target_fs=48000):
    """Get mel division of subbands.

    Reference:
        https://github.com/lucidrains/BS-RoFormer/blob/main/bs_roformer/mel_band_roformer.py#L432-L464

    Args:
        input_dim (int): number of frequency bins corresponding to `target_fs`
            Assumed to be n_fft // 2 + 1, where n_fft is the FFT size.
        n_mels (int): number of mel frequency bands to be used.
        target_fs (int): target sampling frequency in Hz.

    Returns:
        subbands (tuple): a tuple of overlapping subbands (start and end indices)
    """
    assert input_dim % 2 == 1, input_dim
    assert target_fs > 0, target_fs
    n_fft = (input_dim - 1) * 2
    # shape: (n_mels, input_dim)
    mel_filter_bank = librosa.filters.mel(sr=target_fs, n_fft=n_fft, n_mels=n_mels)
    # Make sure the first and last frequency bins are fully covered
    mel_filter_bank[0, 0], mel_filter_bank[-1, -1] = 1.0, 1.0
    freqs_per_band = mel_filter_bank > 0
    if not freqs_per_band.any(axis=0).all():
        raise ValueError(
            "The mel filter bank does not cover the full frequency range. "
            "Please check the input_dim and target_fs."
        )
    # Convert mel filter bank to subbands
    subbands = []
    for i in range(n_mels):
        diff = np.diff(freqs_per_band[i], prepend=False, append=False)
        indices = np.argwhere(diff).squeeze().tolist()
        if len(indices) != 2:
            raise ValueError(
                "Expected a consecutive range of frequencies (exactly two indices) "
                f"for each mel band, but got {len(indices)} indices for band {i}."
            )
        start, end = indices
        subbands.append((start, end - 1))
    return tuple(subbands)


def get_erb_subbands(input_dim, min_freq_idx=0, n_erbs=64, target_fs=48000):
    """Get Equivalent Rectangular Bandwidth (ERB) division of subbands.

    Reference:
        https://github.com/Xiaobin-Rong/gtcrn/blob/main/stream/gtcrn.py#L11-L49

    Args:
        input_dim (int): number of frequency bins corresponding to `target_fs`
            Assumed to be n_fft // 2 + 1, where n_fft is the FFT size.
        min_freq_idx (int): bin index of the minimum frequency to start the ERB bands.
            Frequency bins below this value will be kept as is.
            `min_freq_idx / input_dim * target_fs / 2` is the minimum frequency in Hz.
        n_erbs (int): number of ERB frequency bands to be used.
        target_fs (int): target sampling frequency in Hz.

    Returns:
        subbands (tuple): a tuple of overlapping subbands (start and end indices)
            len(subbands) = min_freq_idx + n_erbs
    """

    def hz2erb(freq_hz):
        erb_f = 21.4 * np.log10(0.00437 * freq_hz + 1)
        return erb_f

    def erb2hz(erb_f):
        freq_hz = (10 ** (erb_f / 21.4) - 1) / 0.00437
        return freq_hz

    assert input_dim % 2 == 1, input_dim
    assert target_fs > 0, target_fs
    n_fft = (input_dim - 1) * 2
    freqs = torch.fft.rfftfreq(n_fft, 1.0 / target_fs).tolist()

    erb_low = hz2erb(freqs[min_freq_idx])
    erb_high = hz2erb(freqs[-1])
    erb_points = np.linspace(erb_low, erb_high, n_erbs + 1)
    bins = np.round(erb2hz(erb_points) / target_fs * n_fft).astype(np.int32)
    assert bins[0] == min_freq_idx, (min_freq_idx, bins)
    bins[-1] = input_dim  # Ensure the last bin covers the full range
    if np.diff(bins).min() < 1:
        raise ValueError(
            "The ERB filter bank has duplicated frequency bands. "
            "Please increase `min_freq_idx` to ensure unique subbands."
        )
    subbands = [(i, i) for i in range(min_freq_idx)] + [
        (bins[i], bins[i + 1] - 1) for i in range(len(bins) - 1)
    ]
    return tuple(subbands)
