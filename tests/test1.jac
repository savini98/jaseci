import torch;
import torch.nn as nn;
import torch._dynamo as dynamo;
import from torch.fx { GraphModule }
import from torch.fx.passes.graph_drawer { FxGraphDrawer }
import from torch._dynamo.testing { rand_strided }
import from random { randint }
import types;
# import from tools { inspect_backend }
import csv;

import logging;

import os;
import time;


with entry {
    torch._dynamo.config.capture_scalar_outputs = True;
}


# class Model(nn.Module) {
#     def init(self: Model) {
#         <>super().init();
#     }

#     def forward(self: Model, x: Any) {
#         path1 = torch.sin(x);
#         path2 = torch.cos(x);
#         return ((path1 ** 2) + (path2 ** 2));
#     }
# }


# class TracableModel(nn.Module) {
#     def init(self: TracableModel) {
#         <>super().init();
#         self.conv = nn.Conv2d(3, 16, 3, padding=1);
#         self.bn = nn.BatchNorm2d(16);
#         self.relu = nn.ReLU();
#         self.fc = nn.Linear(((16 * 32) * 32), 10);
#     }

#     def forward(self: TracableModel, x: Any) {
#         x = self.conv(x);
#         x = self.bn(x);
#         x = self.relu(x);
#         x = x.view(x.size(0), -1);
#         x = self.fc(x);
#         return x;
#     }
# }


# class ModelWithControlFlow(nn.Module) {
#     def init(self: ModelWithControlFlow) {
#         <>super().init();
#         self.conv = nn.Conv2d(3, 16, 3, padding=1);
#         self.bn = nn.BatchNorm2d(16);
#         self.relu = nn.ReLU();
#         self.fc = nn.Linear(((16 * 32) * 32), 10);
#     }

#     def forward(self: ModelWithControlFlow, x: Any) {
#         x = self.conv(x);
#         x = self.bn(x);
#         if (x.sum() > 0) {
#             x = self.relu(x);
#         }
#             else
#             {
#             x = -x;
#             }

#         x = x.view(x.size(0), -1);
#         x = self.fc(x);
#         return x;
#     }
# }


# class ModelWithPythonOps(nn.Module) {
#     def init(self: ModelWithPythonOps) {
#         <>super().init();
#         self.conv = nn.Conv2d(3, 16, 3, padding=1);
#         self.bn = nn.BatchNorm2d(16);
#         self.relu = nn.ReLU();
#         self.fc = nn.Linear(((16 * 32) * 32), 10);
#     }

#     def forward(self: ModelWithPythonOps, x: Any) {
#         x = self.conv(x);
#         x = self.bn(x);
#         x = self.relu(x);
#         if (x.shape[ 2 ] == 32) {
#             print('32');
#         }
#             else
#             {
#             print('not 32');
#             }

#         x = x.view(x.size(0), -1);
#         x = self.fc(x);
#         return x;
#     }
# }


# with entry {
#     sample_input = torch.randn(2, 3, 32, 32);
#     torch._dynamo.reset();
# }


def with_breaks(a: Any, b: Any) {
    x = (a / (torch.abs(a) + 1));
    b = (b * 1.414);
    if (b.sum() < 0) {
        b = (b * -1);
    } else{
            b = (b * -2);
    }
    
    return (x * b);
}


# def without_breaks_where(a: Any, b: Any) {
#     x = (a / (torch.abs(a) + 1));
#     b = (b * 1.414);
#     y = b.sum();
#     b1 = (b * -1);
#     b2 = (b * -2);
#     b = torch.where((y < 0), b1, b2);
#     return (x * b);
# }


# def without_breaks_cond(a: Any, b: Any) {
#     x = (a / (torch.abs(a) + 1));
#     b = (b * 1.414);
#     def true_fn(b: Any) {
#         return (b * -1);
#     }
#     def false_fn(b: Any) {
#         return (b * -2);
#     }
#     b = torch.cond((b.sum() < 0), true_fn, false_fn, (b, ));
#     return (x * b);
# }


# def fn(x: Any) {
#     my_dict = {'a' : 1 };
#     my_dict[ 'a' ] = x.item();
#     return my_dict[ 'a' ];
# }


with entry {
    a = torch.randn(4, 4, dtype=torch.float32, device='cuda');
    b = torch.randn(4, 4, dtype=torch.float32, device='cuda');
    fn_x = torch.compile(with_breaks, backend='inductor');
    explanation = dynamo.explain(fn_x)(a,b);
    print('Graph breaks:', explanation.graph_break_count);
    print('Graph count:', explanation.graph_count);
    val4 = fn_x(a,b);
}