from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoImageProcessor, AutoModel
torch.backends.cuda.matmul.allow_tf32 = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvBlock(nn.Module):
    """
    Conv2d -> LayerNorm2d -> GELU を2回繰り返すブロック。
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = LayerNorm2d(out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = LayerNorm2d(out_channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        return x


class SAMNeck(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.neck = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            LayerNorm2d(dim, eps=1e-6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.neck(x)


class UpBlock(nn.Module):
    """
    (B, c_in, h, w)
      → upsample (B, c_in, 2h, 2w)
      → 1x1 conv (c_in → c_out)
      → SAMNeck(c_out)
      = (B, c_out, 2h, 2w)
    """
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.conv_reduce = nn.Conv2d(c_in, c_out, kernel_size=1)
        self.sam = SAMNeck(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = F.interpolate(x, size=(h * 2, w * 2), mode="bilinear", align_corners=False)
        x = self.conv_reduce(x)
        x = self.sam(x)
        return x


class UpBlockUNet(nn.Module):
    """
    U-Net の上側ブロック:
    - Bilinear upsample (×2)
    - skip と concat（あれば）
    - ConvBlock
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, scale_factor: float = 2.0):
        """
        Args:
            in_channels:   入力特徴マップのチャネル数（アップサンプル前）
            skip_channels: スキップ接続側のチャネル数 (0 or None なら skip なし)
            out_channels:  ConvBlock 出力チャネル数
        """
        super().__init__()
        self.skip_channels = skip_channels
        self.scale_factor = scale_factor

        if skip_channels is None or skip_channels == 0:
            conv_in_ch = in_channels
        else:
            conv_in_ch = in_channels + skip_channels

        self.block = ConvBlock(conv_in_ch, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)

        if self.skip_channels is not None and self.skip_channels > 0 and skip is not None:
            # spatial サイズが微妙にずれている場合の保険（中心クロップ or interpolate などは必要に応じて追加）
            if x.shape[-2:] != skip.shape[-2:]:
                # ここでは簡易に skip 側を x に合わせる
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.block(x)
        return x


class DinoV3VitCNNUnet(nn.Module):
    """
    ViT (DINOv3) + ConvNeXt (DINOv3) hybrid:
    - ConvNeXt: multi-scale features at H, H/4, H/8, H/16, H/32
    - ViT: patch features at H/16
    - Decoder: H/32 → H/16 → H/8 → H/4 → H/2 → H, with
        concat (CNN + ViT/decoder) → upsample → 1x1 conv (channel reduction) → SAMNeck
    最終出力: (B, nout, H, W)
    """

    def __init__(
        self,
        backbone,
        ps: int = 8,
        nout: int = 3,
        bsize: int = 256,
        rdrop: float = 0.4,
        freeze_backbone: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.ps = ps
        self.nout = nout
        self.bsize = bsize
        self.rdrop = rdrop
        self.dtype = dtype

        # --- backbones ---
        self.vit = AutoModel.from_pretrained(backbone['vit'])
        print(f"vit backbone {backbone['vit']} loaded")
        self.cnn = AutoModel.from_pretrained(backbone['cnn'])
        print(f"cnn backbone {backbone['cnn']} loaded")

        vit_dim = self.vit.config.hidden_size          # e.g. 384 for vits16
        c4, c8, c16, c32 = self.cnn.config.hidden_sizes  # e.g. [96, 192, 384, 768]
        c0 = 3  # input RGB

        # ViT / CNN H/16 projection to a common c16 dim
        self.proj_vit16 = nn.Conv2d(vit_dim, c16, kernel_size=1)
        self.proj_cnn16 = nn.Conv2d(c16, c16, kernel_size=1)

        # UpBlocks:
        # H/32 → H/16 (dec start)
        self.up32_16 = UpBlock(c32, c16)           # 768 → 384

        # H/16 → H/8  (concat: vit16 + f16 + x16dec → 3*c16)
        self.up16_8 = UpBlock(3 * c16, c8)         # 1152 → 192

        # H/8 → H/4   (concat: f8 + x8)
        self.up8_4 = UpBlock(c8 + c8, c4)          # 384 → 96

        # H/4 → H/2   (concat: f4 + x4)
        self.up4_2 = UpBlock(2 * c4, 2 * c0)       # 192 → 6

        # H/2 → H     (concat: f0_down + x2)
        self.up2_full = UpBlock(c0 + 2 * c0, c0)   # 9 → 3

        # final 1x1 conv to nout
        self.out = nn.Conv2d(c0, nout, kernel_size=1)

        # Cellpose 互換のダミーパラメータ
        self.diam_labels = nn.Parameter(torch.tensor([30.]), requires_grad=False)
        self.diam_mean = nn.Parameter(torch.tensor([30.]), requires_grad=False)

        if freeze_backbone:
            print("encoders are frozen")
            for p in self.vit.parameters():
                p.requires_grad = False
            for p in self.cnn.parameters():
                p.requires_grad = False
            self.vit.eval()
            self.cnn.eval()
        else:
            print("encoders are training")

        if self.dtype != torch.float32:
            self.to(self.dtype)

    # ----------------------------
    # ViT: (B, 3, H, W) → (B, C_vit, H/16, W/16)
    # ----------------------------
    def vit_forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        out = self.vit(x)
        tokens = out.last_hidden_state  # (B, 1+4+N, C)
        # DINOv3: CLS + 4 global tokens + patch tokens
        patch_tokens = tokens[:, 5:, :]  # (B, N, C)
        hp, wp = h // 16, w // 16
        patch_tokens = patch_tokens.reshape(b, hp, wp, -1).permute(0, 3, 1, 2).contiguous()
        return patch_tokens  # (B, C_vit, H/16, W/16)

    # ----------------------------
    # ConvNeXt: multi-scale features
    # returns [f0(H), f4(H/4), f8(H/8), f16(H/16), f32(H/32)]
    # ----------------------------
    def cnn_forward(self, x: torch.Tensor):
        out = self.cnn(x, output_hidden_states=True)
        feats = [t for t in out.hidden_states if t.dim() == 4]
        # sort by spatial size descending: H, H/4, H/8, H/16, H/32
        feats = sorted(feats, key=lambda f: f.shape[2], reverse=True)
        # we expect: len(feats) >= 5 and first 5 are the ones we want
        f0, f4, f8, f16, f32 = feats[:5]
        return f0, f4, f8, f16, f32

    # ----------------------------
    # Forward
    # ----------------------------
    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, H, W), H, W は 16 の倍数
        return: (B, nout, H, W), dummy_readout
        """
        b, _, h, w = x.shape

        # CNN pyramid
        f0, f4, f8, f16, f32 = self.cnn_forward(x)

        # ViT feature at H/16
        vit16 = self.vit_forward(x)  # (B, C_vit, H/16, W/16)

        # -------------------------
        # Stage 1: H/32 → H/16
        # -------------------------
        x16_dec = self.up32_16(f32)  # (B, c16, H/16, W/16)

        # -------------------------
        # Stage 2: H/16 → H/8
        # concat: ViT16 + CNN16 + decoder16
        # -------------------------
        vit16_proj = self.proj_vit16(vit16)   # (B, c16, H/16, W/16)
        f16_proj = self.proj_cnn16(f16)       # (B, c16, H/16, W/16)
        x = torch.cat([vit16_proj, f16_proj, x16_dec], dim=1)  # (B, 3*c16, H/16, W/16)
        x8 = self.up16_8(x)  # (B, c8, H/8, W/8)

        # -------------------------
        # Stage 3: H/8 → H/4
        # concat: CNN8 + decoder8
        # -------------------------
        x = torch.cat([f8, x8], dim=1)  # (B, c8 + c8, H/8, W/8)
        x4 = self.up8_4(x)  # (B, c4, H/4, W/4)

        # -------------------------
        # Stage 4: H/4 → H/2
        # concat: CNN4 + decoder4
        # -------------------------
        x = torch.cat([f4, x4], dim=1)  # (B, 2*c4, H/4, W/4)
        x2 = self.up4_2(x)  # (B, 2*c0, H/2, W/2)

        # -------------------------
        # Stage 5: H/2 → H
        # concat: downsampled CNN0 + decoder2
        # -------------------------
        f0_down = F.interpolate(f0, size=(h // 2, w // 2), mode="bilinear", align_corners=False)
        x = torch.cat([f0_down, x2], dim=1)  # (B, c0 + 2*c0, H/2, W/2)
        x_full = self.up2_full(x)  # (B, c0, H, W)

        out = self.out(x_full)  # (B, nout, H, W)

        dummy_readout = torch.zeros((b, 256), device=out.device, dtype=out.dtype)
        return out, dummy_readout

    def load_model(self, PATH, device, strict = False):
        state_dict = torch.load(PATH, map_location = device, weights_only=True)
        keys = [k for k in state_dict.keys()]

        if keys[0][:7] == "module.":
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict, strict = strict)
        else:
            self.load_state_dict(state_dict, strict = strict)

        if self.dtype != torch.float32:
            self = self.to(self.dtype)

    @property
    def device(self):
        """
        Get the device of the model.

        Returns:
            torch.device: The device of the model.
        """
        return next(self.parameters()).device

    def save_model(self, filename):
        """
        Save the model to a file.

        Args:
            filename (str): The path to the file where the model will be saved.
        """
        torch.save(self.state_dict(), filename)


class DinoV3CNNOnlyUnet(nn.Module):
    """
    backbone_vit を使わず、backbone_cnn の特徴量 (C0〜C3) だけで
    U-Net-like なモデルを構成するクラス。

    - デフォルト backbone_cnn: facebook/dinov3-convnext-tiny-pretrain-lvd1689m (ConvNeXt-T 相当を想定)
    - 入力: x (B, 3, H, W), H, W は16の倍数
    - 出力:
        out           : (B, nout, H, W)  ← ロジット（活性化なし）
        dummy_readout : (B, 256)         ← 互換用のダミー
    """
    def __init__(
        self,
        backbone=None,
        nout: int = 3,
        decoder_channels: Optional[List[int]] = None,
        freeze_backbone: bool = False,
        dtype: str = "troch.float32",
    ):
        super().__init__()

        # --- backbone の構築 or 受け取り ---
        self.cnn = AutoModel.from_pretrained(backbone['cnn'])
        print(f"backbone {backbone['cnn']} loaded")

        self.diam_labels = nn.Parameter(torch.tensor([30.]), requires_grad=False)
        self.diam_mean = nn.Parameter(torch.tensor([30.]), requires_grad=False)

        if freeze_backbone:
            for p in self.cnn.parameters():
                p.requires_grad = False

        # ConvNeXt Tiny (DINOv3 convnext tiny) を想定した Encoder 側のチャネル数
        #   C0: 96, C1: 192, C2: 384, C3: 768
        self.encoder_channels = self.backbone.config.hidden_sizes

        # Decoder 側のチャネル数
        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64]
        assert len(decoder_channels) == 4, "decoder_channels は4要素のリストで指定してください。"

        self.decoder_channels = decoder_channels
        self.dtype = dtype

        # --- U-Net Decoder Blocks ---
        # C3 (768) -> up3 -> uses skip C2 (384) -> 512
        self.up3 = UpBlockUNet(
            in_channels=self.encoder_channels[3],  # 768
            skip_channels=self.encoder_channels[2],  # 384
            out_channels=self.decoder_channels[0],  # 512
        )
        # up2: 512 + C1(192) -> 256
        self.up2 = UpBlockUNet(
            in_channels=self.decoder_channels[0],  # 512
            skip_channels=self.encoder_channels[1],  # 192
            out_channels=self.decoder_channels[1],  # 256
        )
        # up1: 256 + C0(96) -> 128
        self.up1 = UpBlockUNet(
            in_channels=self.decoder_channels[1],  # 256
            skip_channels=self.encoder_channels[0],  # 96
            out_channels=self.decoder_channels[2],  # 128
        )
        # up0: 128 -> (H, W), skip なし -> 64
        self.up0 = UpBlockUNet(
            in_channels=self.decoder_channels[2],  # 128
            skip_channels=0,
            out_channels=self.decoder_channels[3],  # 64
            scale_factor=4.0
        )

        # 最終 1×1 Conv で nout へ
        self.final_conv = nn.Conv2d(self.decoder_channels[3], nout, kernel_size=1)

    # ------------------------------------------------------------------
    #  Backbone から C0〜C3 を取り出すためのユーティリティ
    # ------------------------------------------------------------------
    def _forward_backbone_features(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        backbone_cnn から 4段階の特徴マップ (C0, C1, C2, C3) を得るための補助関数。

        プロジェクト環境の DINOv3 ConvNeXt 実装に合わせて、
        必要ならこの関数だけ書き換える。
        """
        # 例1: backbone.forward_features(x) が dict を返す場合
        if hasattr(self.cnn, "forward_features"):
            feats = self.cnn.forward_features(x)
            # 場合分け: dict / list / tuple など
            if isinstance(feats, dict):
                # ここは実装に応じてキー名を変えてください
                # 例: timm ConvNeXt の features_only=True なら ["0","1","2","3"] など
                # 仮に DINOv3 convnext が "res2"〜"res5" を返すとした例:
                candidate_keys = [
                    ("res2", "res3", "res4", "res5"),
                    ("0", "1", "2", "3"),
                    ("stage0", "stage1", "stage2", "stage3"),
                ]
                for keys in candidate_keys:
                    if all(k in feats for k in keys):
                        c0, c1, c2, c3 = (feats[keys[0]], feats[keys[1]], feats[keys[2]], feats[keys[3]])
                        return c0, c1, c2, c3
                raise RuntimeError(
                    "forward_features の返り値 dict から C0〜C3 を特定できませんでした。"
                )
            elif isinstance(feats, (list, tuple)):
                # list/tuple の場合: 先頭4つを C0〜C3 とみなす
                if len(feats) < 4:
                    raise RuntimeError(
                        f"backbone.forward_features(x) が {len(feats)} 個しか特徴を返しません。少なくとも4段必要です。"
                    )
                c0, c1, c2, c3 = feats[0], feats[1], feats[2], feats[3]
                return c0, c1, c2, c3
            else:
                # 単一 Tensor を返すなど
                raise RuntimeError(
                    "backbone.forward_features(x) が dict/list/tuple 以外を返しました。"
                    "プロジェクトに合わせて _forward_backbone_features を実装し直してください。"
                )

        # 例2: backbone(x) がそのまま特徴リストを返す場合
        out = self.cnn(x, output_hidden_states=True)
        feats = [t for t in out.hidden_states if t.dim() == 4]
        # sort by spatial size descending: H, H/4, H/8, H/16, H/32
        feats = sorted(feats, key=lambda f: f.shape[2], reverse=True)
        # we expect: len(feats) >= 5 and first 5 are the ones we want
        f4, f8, f16, f32 = feats[1:5]
        return f4, f8, f16, f32

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W), H, W は 16 の倍数を想定

        Returns:
            out          : (B, nout, H, W)  # ロジット
            dummy_readout: (B, 256)
        """
        b, c, h, w = x.shape

        # Backbone で C0〜C3 を取得
        c0, c1, c2, c3 = self._forward_backbone_features(x)
        # 期待形状（ConvNeXt Tiny の場合）:
        #   c0: (B,  96, H/4,  W/4)
        #   c1: (B, 192, H/8,  W/8)
        #   c2: (B, 384, H/16, W/16)
        #   c3: (B, 768, H/32, W/32)

        # Decoder
        h_dec = self.up3(c3, c2)   # -> (B, 512, H/16, W/16)
        h_dec = self.up2(h_dec, c1)  # -> (B, 256, H/8,  W/8)
        h_dec = self.up1(h_dec, c0)  # -> (B, 128, H/4,  W/4)
        h_dec = self.up0(h_dec, None)  # -> (B, 64,  H,   W)

        out = self.final_conv(h_dec)   # (B, nout, H, W)

        # dummy_readout: (B, 256)
        dummy_readout = torch.zeros(
            (b, 256),
            device=out.device,
            dtype=out.dtype,
        )

        return out, dummy_readout
    
    @property
    def device(self):
        """
        Get the device of the model.

        Returns:
            torch.device: The device of the model.
        """
        return next(self.parameters()).device

    def save_model(self, filename):
        """
        Save the model to a file.

        Args:
            filename (str): The path to the file where the model will be saved.
        """
        torch.save(self.state_dict(), filename)
