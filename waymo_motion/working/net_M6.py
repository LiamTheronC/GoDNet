# M6 net

import numpy as np
from fractions import gcd
from numbers import Number
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from utils import to_long, gpu
from memory_profiler import profile



class Linear(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32, act=True):
        super(Linear, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.linear = nn.Linear(n_in, n_out, bias=False)
        
        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')
        
        self.relu = nn.ReLU(inplace=True)
        #self.relu = nn.Sigmoid()
        self.act = act

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out


class LinearRes(nn.Module):
    def __init__(self, n_in, n_out, norm='GN', ng=32):
        super(LinearRes, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.linear1 = nn.Linear(n_in, n_out, bias=False)
        self.linear2 = nn.Linear(n_out, n_out, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if norm == 'GN':
            self.norm1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.norm2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm1 = nn.BatchNorm1d(n_out)
            self.norm2 = nn.BatchNorm1d(n_out)
        else:   
            exit('SyncBN has not been added!')

        if n_in != n_out:
            if norm == 'GN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.transform = nn.Sequential(
                    nn.Linear(n_in, n_out, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.transform = None

    def forward(self, x):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm2(out)

        if self.transform is not None:
            out += self.transform(x)
            
        else:
            out += x
        out = self.relu(out) 

        return out


class Conv1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Conv1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=(int(kernel_size) - 1) // 2, stride=stride, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out
 

class Res1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Res1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace = True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.bn1 = nn.BatchNorm1d(n_out)
            self.bn2 = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        if stride != 1 or n_out != n_in:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                        nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                        nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.downsample = nn.Sequential(
                        nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out


def actor_gather(actors: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
    """
    actors is data['feat']
     
    """

    batch_size = len(actors)
    num_actors = [len(x) for x in actors]

    actors = [torch.stack(x).transpose(1, 2) for x in actors]
    actors = torch.cat(actors, 0)

    actor_idcs = []
    count = 0
    for i in range(batch_size):
        idcs = torch.arange(count, count + num_actors[i]).to(actors.device)
        actor_idcs.append(idcs)
        count += num_actors[i]
    return actors, actor_idcs


def graph_gather(graphs):
    batch_size = len(graphs)
    node_idcs = []
    count = 0
    counts = []

    for i in range(batch_size):
        counts.append(count)
        idcs = torch.arange(count, count + graphs[i]["num_nodes"])
        node_idcs.append(idcs)
        count = count + graphs[i]["num_nodes"]

    graph = dict()
    graph["idcs"] = node_idcs
    graph["ctrs"] = [x["ctrs"] for x in graphs]

    graph['feats'] = torch.cat([x['feats'] for x in graphs], 0)

    for k1 in ["pre", "suc"]:
        graph[k1] = []
        for i in range(len(graphs[0]["pre"])):
            graph[k1].append(dict())
            for k2 in ["u", "v"]:
                graph[k1][i][k2] = torch.cat(
                    [graphs[j][k1][i][k2] + counts[j] for j in range(batch_size)], 0
                )

    for k1 in ["left", "right"]:
        graph[k1] = dict()
        for k2 in ["u", "v"]:
            temp = [graphs[i][k1][k2] + counts[i] for i in range(batch_size)]
            temp = [
                x if x.dim() > 0 else graph["pre"][0]["u"].new().resize_(0)
                for x in temp
            ]
            graph[k1][k2] = torch.cat(temp)
    
    return graph


class ActorNet(nn.Module):
    def __init__(self,config) -> None:
        super(ActorNet,self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_in = 4
        n_out = [32, 64, 128]
        blocks = [Res1d, Res1d, Res1d]
        num_blocks = [2, 2, 2]

        groups = []

        for i in range(len(num_blocks)):

            group = []

            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))

            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            
            groups.append(nn.Sequential(*group))
            
            n_in = n_out[i]

        self.groups = nn.ModuleList(groups)

        n = config['n_actornet']#128
        
        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], n, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)

        self.outlayer = Res1d(n, n, norm=norm, ng=ng)

    def forward(self, actors: Tensor) -> Tensor:
        #actors [batch_size,feature_dim(4),time_step(11)]
        
        out = actors

        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)
        
        out = self.lateral[-1](outputs[-1])

        for i in range(len(outputs) - 2, -1, -1):

            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            tmp = self.lateral[i](outputs[i])

            if out.shape != tmp.shape:
                out = out[:,:,:tmp.shape[2]]

            out += tmp
        
        out = self.outlayer(out)[:,:,-1]

        return out


class MapNet(nn.Module):
  
    def __init__(self, config):
        super(MapNet, self).__init__()
        self.config = config
        n_map = 128
        norm = "GN"
        ng = 1

        self.input = nn.Sequential(
            nn.Linear(3, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )
        self.seg = nn.Sequential(
            nn.Linear(3, n_map),
            nn.ReLU(inplace=True),
            Linear(n_map, n_map, norm=norm, ng=ng, act=False),
        )

        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(ng, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(n_map, n_map, norm=norm, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, graph):
        if (
            len(graph["feats"]) == 0
            or len(graph["pre"][-1]["u"]) == 0
            or len(graph["suc"][-1]["u"]) == 0
        ):
            temp = graph["feats"]
            return (
                temp.new().resize_(0),
                [temp.new().long().resize_(0) for x in graph["node_idcs"]],
                temp.new().resize_(0),
            )

        ctrs = torch.cat(graph["ctrs"], 0)
        feat = self.input(ctrs)
        feat += self.seg(graph["feats"])
        feat = self.relu(feat)

        """fuse map"""
        res = feat
        for i in range(len(self.fuse["ctr"])):
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1 = key[:3]
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        graph[k1][k2]["u"],
                        self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                    )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"],
                    self.fuse["left"][i](feat[graph["left"]["v"]]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"],
                    self.fuse["right"][i](feat[graph["right"]["v"]]),
                )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat

        return feat , graph["idcs"], graph["ctrs"]


class Att(nn.Module):
    def __init__(self, n_agt: int, n_ctx: int) -> None:
        super(Att, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(3, n_ctx),
            nn.ReLU(inplace=True),
            Linear(n_ctx, n_ctx, norm=norm, ng=ng),
        )

        self.query = Linear(n_agt, n_ctx, norm=norm, ng=ng)

        self.ctx = nn.Sequential(
            Linear(3 * n_ctx, n_agt, norm=norm, ng=ng),
            nn.Linear(n_agt, n_agt, bias=False),
        )

        self.agt = nn.Linear(n_agt, n_agt, bias=False)
        self.norm = nn.GroupNorm(gcd(ng, n_agt), n_agt)
        self.linear = Linear(n_agt, n_agt, norm=norm, ng=ng, act=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, agts: Tensor, agt_idcs: List[Tensor], agt_ctrs: List[Tensor], ctx: Tensor, ctx_idcs: List[Tensor], ctx_ctrs: List[Tensor], dist_th: float) -> Tensor:
        # feat, graph["idcs"], graph["ctrs"], actors, actor_idcs, actor_ctrs, config["actor2map_dist"]      
        res = agts
        if len(ctx) == 0:
            agts = self.agt(agts)
            agts = self.relu(agts)
            agts = self.linear(agts)
            agts += res
            agts = self.relu(agts)
            return agts

        batch_size = len(agt_idcs)
        hi, wi = [], []
        hi_count, wi_count = 0, 0

        
        for i in range(batch_size):
        
            dist = agt_ctrs[i].view(-1, 1, 3) - ctx_ctrs[i].view(1, -1, 3)
            dist = torch.sqrt((dist ** 2).sum(2))
            mask = dist <= dist_th

            idcs = torch.nonzero(mask, as_tuple=False)
            if len(idcs) == 0:
                continue

            hi.append(idcs[:, 0] + hi_count)
            wi.append(idcs[:, 1] + wi_count)
            hi_count += len(agt_idcs[i])
            wi_count += len(ctx_idcs[i])

        if hi == []:
            print('WARNING!!! - Attention')

        hi = torch.cat(hi, 0)
        wi = torch.cat(wi, 0)

        agt_ctrs = torch.cat(agt_ctrs, 0)
        ctx_ctrs = torch.cat(ctx_ctrs, 0)
        dist = agt_ctrs[hi] - ctx_ctrs[wi]
        dist = self.dist(dist)

        query = self.query(agts[hi])

        ctx = ctx[wi]
        ctx = torch.cat((dist, query, ctx), 1)
        ctx = self.ctx(ctx)

        agts = self.agt(agts)
        agts.index_add_(0, hi, ctx)
        agts = self.norm(agts)
        agts = self.relu(agts)

        agts = self.linear(agts)
        agts += res
        agts = self.relu(agts)

        return agts


class A2M(nn.Module):
    """
    Actor to Map Fusion:  fuses real-time traffic information from
    actor nodes to lane nodes
    """
    def __init__(self, config):
        super(A2M, self).__init__()
        self.config = config
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        """fuse meta, static, dyn"""
        self.meta = Linear(n_map, n_map, norm=norm, ng=ng)
        att = []
        for i in range(2):
            att.append(Att(n_map, config["n_actor"]))
        self.att = nn.ModuleList(att)

    def forward(self, feat: Tensor, graph: Dict[str, Union[List[Tensor], Tensor, List[Dict[str, Tensor]], Dict[str, Tensor]]], actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Tensor:
        """meta, static and dyn fuse using attention"""
        
        feat = self.meta(feat)

        for i in range(len(self.att)):
            feat = self.att[i](
                feat,
                graph["idcs"],
                graph["ctrs"],
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2map_dist"],
            )
        return feat


class M2M(nn.Module):
 
    def __init__(self, config):
        super(M2M, self).__init__()
        self.config = config
        n_map = config["n_map"]
        norm = "GN"
        ng = 1

        keys = ["ctr", "norm", "ctr2", "left", "right"]
        for i in range(config["num_scales"]):
            keys.append("pre" + str(i))
            keys.append("suc" + str(i))

        fuse = dict()
        for key in keys:
            fuse[key] = []

        for i in range(4):
            for key in fuse:
                if key in ["norm"]:
                    fuse[key].append(nn.GroupNorm(gcd(ng, n_map), n_map))
                elif key in ["ctr2"]:
                    fuse[key].append(Linear(n_map, n_map, norm=norm, ng=ng, act=False))
                else:
                    fuse[key].append(nn.Linear(n_map, n_map, bias=False))

        for key in fuse:
            fuse[key] = nn.ModuleList(fuse[key])
        self.fuse = nn.ModuleDict(fuse)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat: Tensor, graph: Dict) -> Tensor:
        """fuse map"""
        res = feat
        for i in range(len(self.fuse["ctr"])):
            temp = self.fuse["ctr"][i](feat)
            for key in self.fuse:
                if key.startswith("pre") or key.startswith("suc"):
                    k1 = key[:3]
                    k2 = int(key[3:])
                    temp.index_add_(
                        0,
                        graph[k1][k2]["u"],
                        self.fuse[key][i](feat[graph[k1][k2]["v"]]),
                    )

            if len(graph["left"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["left"]["u"],
                    self.fuse["left"][i](feat[graph["left"]["v"]]),
                )
            if len(graph["right"]["u"] > 0):
                temp.index_add_(
                    0,
                    graph["right"]["u"],
                    self.fuse["right"][i](feat[graph["right"]["v"]]),
                )

            feat = self.fuse["norm"][i](temp)
            feat = self.relu(feat)

            feat = self.fuse["ctr2"][i](feat)
            feat += res
            feat = self.relu(feat)
            res = feat
        return feat


class M2A(nn.Module):
    """
    The lane to actor block fuses updated
        map information from lane nodes to actor nodes
    """
    def __init__(self, config):
        super(M2A, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]
        n_map = config["n_map"]

        att = []
        for i in range(2):
            att.append(Att(n_actor, n_map))
        self.att = nn.ModuleList(att)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor], nodes: Tensor, node_idcs: List[Tensor], node_ctrs: List[Tensor]) -> Tensor:
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                nodes,
                node_idcs,
                node_ctrs,
                self.config["map2actor_dist"],
            )
        return actors


class A2A(nn.Module):
    """
    The actor to actor block performs interactions among actors.
    """
    def __init__(self, config):
        super(A2A, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]
        n_map = config["n_map"]

        att = []
        for i in range(2):
            att.append(Att(n_actor, n_actor))
        self.att = nn.ModuleList(att)

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Tensor:
        for i in range(len(self.att)):
            actors = self.att[i](
                actors,
                actor_idcs,
                actor_ctrs,
                actors,
                actor_idcs,
                actor_ctrs,
                self.config["actor2actor_dist"],
            )
        return actors


class PredNet(nn.Module):
    def __init__(self, config):
        super(PredNet, self).__init__()
        self.config = config
        norm = "GN"
        ng = 1

        n_actor = config["n_actor"]

        pred = []
        for i in range(config["num_mods"]):
            pred.append(
                nn.Sequential(
                    LinearRes(n_actor, n_actor, norm=norm, ng=ng),
                    nn.Linear(n_actor, 2 * config["num_preds"]),
                )
            )
        self.pred = nn.ModuleList(pred)

        self.att_dest = AttDest(n_actor)
        self.cls = nn.Sequential(
            LinearRes(n_actor, n_actor, norm=norm, ng=ng), nn.Linear(n_actor, 1)
        )

    def forward(self, actors: Tensor, actor_idcs: List[Tensor], actor_ctrs: List[Tensor]) -> Dict[str, List[Tensor]]:
        preds = []
        for i in range(len(self.pred)):
            preds.append(self.pred[i](actors))
        reg = torch.cat([x.unsqueeze(1) for x in preds], 1)
        reg = reg.view(reg.size(0), reg.size(1), -1, 2)

        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            ctrs = actor_ctrs[i].view(-1, 1, 1, 3)
            reg[idcs] = reg[idcs] + ctrs[:,:,:,:2]

        dest_ctrs = reg[:, :, -1].detach()
        feats = self.att_dest(actors, torch.cat(actor_ctrs, 0), dest_ctrs)
        cls = self.cls(feats).view(-1, self.config["num_mods"])

        cls, sort_idcs = cls.sort(1, descending=True)
        row_idcs = torch.arange(len(sort_idcs)).long().to(sort_idcs.device)
        row_idcs = row_idcs.view(-1, 1).repeat(1, sort_idcs.size(1)).view(-1)
        sort_idcs = sort_idcs.view(-1)
        reg = reg[row_idcs, sort_idcs].view(cls.size(0), cls.size(1), -1, 2)

        out = dict()
        out["cls"], out["reg"] = [], []
        for i in range(len(actor_idcs)):
            idcs = actor_idcs[i]
            out["cls"].append(cls[idcs])
            out["reg"].append(reg[idcs])
        return out


class AttDest(nn.Module):
    def __init__(self, n_agt: int):
        super(AttDest, self).__init__()
        norm = "GN"
        ng = 1

        self.dist = nn.Sequential(
            nn.Linear(2, n_agt),
            nn.ReLU(inplace=True),
            Linear(n_agt, n_agt, norm=norm, ng=ng),
        )

        self.agt = Linear(2 * n_agt, n_agt, norm=norm, ng=ng)

    def forward(self, agts: Tensor, agt_ctrs: Tensor, dest_ctrs: Tensor) -> Tensor:
        n_agt = agts.size(1)
        num_mods = dest_ctrs.size(1)

        dist = (agt_ctrs[:,:2].unsqueeze(1) - dest_ctrs).view(-1, 2)
        dist = self.dist(dist)
        agts = agts.unsqueeze(1).repeat(1, num_mods, 1).view(-1, n_agt)

        agts = torch.cat((dist, agts), 1)
        agts = self.agt(agts)
        return agts


class GreatNet(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()

        self.config = config

        self.actor_net = ActorNet(config)
        self.map_net = MapNet(config)

        self.a2m = A2M(config)
        self.m2m = M2M(config)
        self.m2a = M2A(config)
        self.a2a = A2A(config)
        
        self.pred_net = PredNet(config)
    
    def forward(self, data: Dict) -> Tensor:

        actors, actor_idcs = actor_gather(data["feats"])
        actor_ctrs = [torch.stack(i,0) for i in data["ctrs"]]

        actors = gpu(actors)
        actor_idcs = gpu(actor_idcs)
        actor_ctrs = gpu(actor_ctrs)

        actors = self.actor_net(actors)

        #------------------------------------------------------------#

        graph = to_long(data['graph'])
        graph = graph_gather(graph)

        graph = gpu(graph)

        nodes, node_idcs, node_ctrs = self.map_net(graph)

        #------------------------------------------------------------#
        
        nodes = self.a2m(nodes, graph, actors, actor_idcs, actor_ctrs)
        nodes = self.m2m(nodes, graph)
        actors = self.m2a(actors, actor_idcs, actor_ctrs, nodes, node_idcs, node_ctrs)
        actors = self.a2a(actors, actor_idcs, actor_ctrs)
        
        out = self.pred_net(actors, actor_idcs, actor_ctrs)

        return out


class PredLoss(nn.Module):
    def __init__(self, config):
        super(PredLoss, self).__init__()
        self.config = config
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out: Dict[str, List[Tensor]], data) -> Dict[str, Union[Tensor, int]]:
        cls, reg = out["cls"], out["reg"]
        cls = torch.cat([x for x in cls], 0)
        reg = torch.cat([x for x in reg], 0)
        has_preds = pre_gather(data['has_preds']).cuda()
        gt_preds = pre_gather(data['gt2_preds']).float()[:,:,:2].cuda()

        loss_out = dict()
        zero = 0.0 * (cls.sum() + reg.sum())
        loss_out["cls_loss"] = zero.clone()
        loss_out["num_cls"] = 0
        loss_out["reg_loss"] = zero.clone()
        loss_out["num_reg"] = 0

        num_mods, num_preds = self.config["num_mods"], self.config["num_preds"]
        # assert(has_preds.all())

        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(
            has_preds.device
        ) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0

        cls = cls[mask]
        reg = reg[mask]
        gt_preds = gt_preds[mask]
        has_preds = has_preds[mask]
        last_idcs = last_idcs[mask]

        row_idcs = torch.arange(len(last_idcs)).long().to(last_idcs.device)
        dist = []
        for j in range(num_mods):
            dist.append(
                torch.sqrt(
                    (
                        (reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs])
                        ** 2
                    ).sum(1)
                )
            )
        dist = torch.cat([x.unsqueeze(1) for x in dist], 1)
        min_dist, min_idcs = dist.min(1)
        row_idcs = torch.arange(len(min_idcs)).long().to(min_idcs.device)

        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        coef = self.config["cls_coef"]
        loss_out["cls_loss"] += coef * (
            self.config["mgn"] * mask.sum() - mgn[mask].sum()
        )
        loss_out["num_cls"] += mask.sum().item()

        reg = reg[row_idcs, min_idcs]
        coef = self.config["reg_coef"]
        loss_out["reg_loss"] += coef * self.reg_loss(
            reg[has_preds], gt_preds[has_preds]
        )
        loss_out["num_reg"] += has_preds.sum().item()
        loss_out['reg'] = reg
        return loss_out


class Loss(nn.Module):
    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.pred_loss = PredLoss(config)

    def forward(self, out: Dict, data: Dict) -> Dict:
        loss_out = self.pred_loss(out, data)
        loss_out["loss"] = loss_out["cls_loss"] / (
            loss_out["num_cls"] + 1e-10
        ) + loss_out["reg_loss"] / (loss_out["num_reg"] + 1e-10)
        return loss_out
    

def pre_gather(gts: List) -> Tensor:
    tmp = list()
    for g in gts:
        tmp += g
    
    tmp = torch.stack(tmp)

    return tmp


#test

