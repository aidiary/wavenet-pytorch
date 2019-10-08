import math
import torch
import torch.nn.functional as F


class WaveNet(torch.nn.Module):

    def __init__(self, n_in_channels=256, n_layers=16, max_dilation=128,
                 n_residual_channels=64, n_skip_channels=256, n_out_channels=256,
                 n_cond_channels=80, upsamp_window=800, upsamp_stride=200):
        super(WaveNet, self).__init__()

        self.n_layers = n_layers  # 論文ではK
        self.max_dilation = max_dilation
        self.n_residual_channels = n_residual_channels
        self.n_out_channels = n_out_channels

        # WaveNet Vocoderの条件付けとなるメルスペクトログラム（単位: フレーム）を
        # 波形（単位: サンプル）に合うように拡張するのに用いる
        # 論文によるとrepeatするよりupsampleがよい
        self.upsample = torch.nn.ConvTranspose1d(n_cond_channels,
                                                 n_cond_channels,
                                                 upsamp_window,
                                                 upsamp_stride)

        # 条件付けとなるメルスペクトログラムを n_layers 回繰り返す
        # 各レイヤに入力するためにチャネル数を増やす
        # 2を掛けているのはtanh/sigmoidで分離するため
        self.cond_layers = Conv(n_cond_channels,
                                2 * n_residual_channels * n_layers,
                                w_init_gain='tanh')

        # 波形をConv1Dで処理できる3Dテンソル (batch, channel, seq_len) にするため
        self.embed = torch.nn.Embedding(n_in_channels, n_residual_channels)

        # 以下の3つのモジュールは n_layers回繰り返すためリストで管理する
        self.dilate_layers = torch.nn.ModuleList()
        self.res_layers = torch.nn.ModuleList()
        self.skip_layers = torch.nn.ModuleList()

        loop_factor = math.floor(math.log2(max_dilation)) + 1
        for i in range(n_layers):
            dilation = 2 ** (i % loop_factor)

            # Causal Dilated Convolution
            # WaveNetではdilationだけ離れた2つのペアでConvするので
            # kernel_sizeは常に2
            # is_causal=Trueにすると系列の先頭にpaddingされ
            # 過去のサンプルをConvしないのと同じになる
            in_layer = Conv(n_residual_channels, 2 * n_residual_channels,
                            kernel_size=2, dilation=dilation,
                            w_init_gain='tanh', is_causal=True)
            self.dilate_layers.append(in_layer)

            # 1x1 Conv
            # 最後のループのみ不要
            if i < n_layers - 1:
                res_layer = Conv(n_residual_channels,
                                 n_residual_channels,
                                 kernel_size=1,
                                 stride=1,
                                 w_init_gain='linear')
                self.res_layers.append(res_layer)

            # Skip connectionsのチャネル数に合わせるためのConv
            skip_layer = Conv(n_residual_channels,
                              n_skip_channels,
                              kernel_size=1,
                              stride=1,
                              w_init_gain='relu')
            self.skip_layers.append(skip_layer)

        # Skip Connectionsで使う2つの 1x1 Conv
        self.conv_out = Conv(n_skip_channels,
                             n_out_channels,
                             bias=False,
                             w_init_gain='relu')

        # 出力は波形をquantizeした256クラスの値
        # CrossEntropyでsoftmaxが取られるためLinearのままでOK
        self.conv_end = Conv(n_out_channels,
                             n_out_channels,
                             bias=False,
                             w_init_gain='linear')

    def forward(self, features, forward_input):
        # featuresは条件付けされるメルスペクトログラム
        # forward_inputは音声波形（auto-regressive）
        cond_input = self.upsample(features)

        # 音声波形の長さより長くなるようにupsamplingされる必要がある
        assert(cond_input.size(2) >= forward_input.size(1))

        # 長すぎる場合は切る
        if cond_input.size(2) > forward_input.size(1):
            cond_input = cond_input[:, :, :forward_input.size(1)]

        # 波形をConv1Dで処理できる3Dテンソル (batch, channels, seq_len) にする
        forward_input = self.embed(forward_input.long())
        forward_input = forward_input.transpose(1, 2)

        # 条件付けとなるメルスペクトログラムを n_layers 回繰り返す
        # 各レイヤに入力するためにチャネル数を増やす
        # レイヤごとに条件付けが異なるので注意（同じではだめ？）
        # (batch, n_layers, channels, seq_len)
        cond_acts = self.cond_layers(cond_input)
        cond_acts = cond_acts.view(
            cond_acts.size(0), self.n_layers, -1, cond_acts.size(2))

        # Causal Dilated Convolutionのループ
        for i in range(self.n_layers):
            in_act = self.dilate_layers[i](forward_input)

            # メルスペクトログラムによる条件を加える
            in_act = in_act + cond_acts[:, i, :, :]

            # Gated Activation Unit
            # 入力の前半をtanh、後半をsigmoidに入力する
            t_act = torch.tanh(in_act[:, :self.n_residual_channels, :])
            s_act = torch.sigmoid(in_act[:, self.n_residual_channels:, :])
            acts = t_act * s_act

            # 1x1 Conv
            if i < len(self.res_layers):
                res_acts = self.res_layers[i](acts)

            # Residual Connection
            forward_input = res_acts + forward_input

            # 横に抜けるSkip Connections
            # 初回のみ値がないのでそのままセット
            # 2回目のループ以降は出力を足し合わせる
            # TODO: 1x1 Convを通す前の値を入力する？
            if i == 0:
                output = self.skip_layers[i](acts)
            else:
                output = self.skip_layers[i](acts) + output

        # Skip Connectionsの残りの処理
        output = F.relu(output, inplace=True)
        output = self.conv_out(output)
        output = F.relu(output, inplace=True)
        output = self.conv_end(output)

        # 入力の1サンプル先を予測したいため1サンプルだけ右にシフトする
        # 最初には0を入れる
        last = output[:, :, -1]
        last = last.unsqueeze(2)
        output = output[:, :, :-1]
        first = last * 0.0
        output = torch.cat((first, output), dim=2)

        return output


class Conv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 dilation=1, bias=True, w_init_gain='linear', is_causal=False):
        super(Conv, self).__init__()
        self.is_causal = is_causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    dilation=dilation, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        if self.is_causal:
            # Causal Convolution
            # 系列の先頭のみpaddingして過去のサンプル点が使われないようにする
            padding = (int((self.kernel_size - 1) * self.dilation), 0)
            signal = torch.nn.functional.pad(signal, padding)
        return self.conv(signal)


if __name__ == "__main__":
    model = WaveNet()
    features = torch.rand(8, 80, 81)
    forward_input = torch.rand(8, 16000)
    y_pred = model(features, forward_input)
    print(y_pred.shape)
