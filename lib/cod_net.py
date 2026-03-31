import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.pvtv2 import pvt_v2_b2_

# =========================
# BLOQUES BASE
# =========================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return x + self.act(self.norm(self.conv(x)))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.project(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


# =========================
# ATENCIÓN CBAM
# =========================

#https://github.com/Jongchan/attention-module/blob/c06383c514ab0032d044cc6fcd8c8207ea222ea7/MODELS/cbam.py
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

# =========================
# FUSIÓN RGB + TÉRMICA
# =========================

class FeatureAggregation(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)
        self.conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, feats):
        x = torch.cat(feats, dim=1)
        return self.conv3(self.conv1(x))


class GatedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate_rgb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.gate_th = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        # Agregamos CBAM después de la concatenación/fusión
        self.fuse = nn.Conv2d(channels * 2, channels, 1)
        self.cbam = CBAM(channels) 

    def forward(self, rgb, th):
        rgb_att = rgb * self.gate_rgb(rgb)
        th_att = th * self.gate_th(th)
        fused = self.fuse(torch.cat([rgb_att, th_att], dim=1))
        return self.cbam(fused)
        
# =========================
# DECODER
# =========================

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.aggr = FeatureAggregation(in_ch + skip_ch, out_ch)
        self.conv = ConvBlock(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = self.aggr([x, skip])
        return self.conv(x)


# =========================
# CONTORNOS (EDGE HEAD)
# =========================

class EdgeHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels // 4, 1, 1)
        )

    def forward(self, x):
        return self.net(x)


class BoundaryRefinement(nn.Module):
    def forward(self, seg, edge):
        return seg * (1 + torch.sigmoid(edge))


# =========================
# MODELO COMPLETO
# =========================

class CamouflageDetectionNet(nn.Module):
    def __init__(self, features=[64, 128, 256, 512], pretrained=True):
        super().__init__()

        self.backbone = pvt_v2_b2_()
        if pretrained:
            self._load_backbone_weights(
                "C:/Respaldo/Henry/Proyecto Camuflaje/Codigo/AVNet-v2-main/pretrained_pvt/pvt_v2_b2.pth"
            )

        backbone_ch = [64, 128, 320, 512]

        self.rgb_enc = nn.ModuleList([
            ConvBlock(backbone_ch[i], features[i]) for i in range(4)
        ])
        self.th_enc = nn.ModuleList([
            ConvBlock(backbone_ch[i], features[i]) for i in range(4)
        ])

        self.fusions = nn.ModuleList([
            GatedFusion(features[i]) for i in range(4)
        ])

        self.dec3 = DecoderBlock(features[3], features[2], features[2])
        self.dec2 = DecoderBlock(features[2], features[1], features[1])
        self.dec1 = DecoderBlock(features[1], features[0], features[0])

        self.final_conv = ConvBlock(features[0], features[0])

        # Deep supervision
        self.seg_heads = nn.ModuleList([
            nn.Conv2d(features[2], 1, 1),
            nn.Conv2d(features[1], 1, 1),
            nn.Conv2d(features[0], 1, 1),
            nn.Conv2d(features[0], 1, 1),
        ])

        # Contornos
        self.edge_head = EdgeHead(features[0])
        self.boundary_refine = BoundaryRefinement()

    def forward(self, x_rgb, x_th):
        rgb_feats = self.backbone.forward_features(x_rgb)
        th_feats = self.backbone.forward_features(x_th)

        rgb_feats = [e(f) for e, f in zip(self.rgb_enc, rgb_feats)]
        th_feats = [e(f) for e, f in zip(self.th_enc, th_feats)]

        fused = [f(r, t) for f, r, t in zip(self.fusions, rgb_feats, th_feats)]

        d3 = self.dec3(fused[3], fused[2])
        d2 = self.dec2(d3, fused[1])
        d1 = self.dec1(d2, fused[0])

        d0 = self.final_conv(d1)

        # Segmentaciones intermedias
        out3 = F.interpolate(self.seg_heads[0](d3), x_rgb.shape[2:], mode='bilinear', align_corners=False)
        out2 = F.interpolate(self.seg_heads[1](d2), x_rgb.shape[2:], mode='bilinear', align_corners=False)
        out1 = F.interpolate(self.seg_heads[2](d1), x_rgb.shape[2:], mode='bilinear', align_corners=False)
        out0 = F.interpolate(self.seg_heads[3](d0), x_rgb.shape[2:], mode='bilinear', align_corners=False)

        final_out = (out0 + out1 + out2 + out3) / 4

        # Contornos
        edge = self.edge_head(d0)
        edge = F.interpolate(edge, x_rgb.shape[2:], mode='bilinear', align_corners=False)

        # Refinamiento
        final_out = self.boundary_refine(final_out, edge)

        return [out0, out1, out2, out3], final_out, edge

    def _load_backbone_weights(self, path):
        try:
            self.backbone.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
            print("✅ Backbone cargado")
        except Exception as e:
            print("❌ Error:", e)


# =========================
# TEST RÁPIDO
# =========================

if __name__ == "__main__":
    model = CamouflageDetectionNet().cuda()
    x = torch.randn(1, 3, 256, 256).cuda()
    y = torch.randn(1, 3, 256, 256).cuda()

    outs, mask, edge = model(x, y)
    print(mask.shape, edge.shape)
