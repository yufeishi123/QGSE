import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from mmcv.runner import BaseModule
from ..builder import VSEM


@VSEM.register_module()
class VSEMModule(BaseModule):
    def __init__(self, model="ViT-B/32", templates="a photo of a {}"):
        super(VSEMModule, self).__init__()

        self.ALL_CLASSES_SPLIT1={0:'aeroplane', 1:'bicycle', 2:'boat', 3:'bottle', 4:'car', 5:'cat',
                        6:'chair', 7:'diningtable', 8:'dog', 9:'horse',
                        10:'person', 11:'pottedplant', 12:'sheep', 13:'train',
                        14:'tvmonitor', 15:'bird',16:'bus',17:'cow', 18:'motorbike',19: 'sofa'}

        self.BASE_CLASSES_SPLIT1 = {0:'aeroplane', 1:'bicycle', 2:'boat', 3:'bottle', 4:'car',
                               5:'cat', 6:'chair', 7:'diningtable', 8:'dog', 9:'horse',
                               10:'person', 11:'pottedplant', 12:'sheep', 13:'train',
                               14:'tvmonitor'}
        self.device = torch.cuda.current_device()
        self.clip, _ = clip.load(model, device=self.device)
        self.prompts = clip.tokenize([
            templates.format(self.BASE_CLASSES_SPLIT1[idx])
            for idx in range(len(self.BASE_CLASSES_SPLIT1))
        ]).to(self.device)
        with torch.no_grad():
            text_features = self.clip.encode_text(self.prompts)
            self.text_features = F.normalize(text_features, dim=-1, p=2)
            self.text_features = text_features.to(torch.float32)
        # self.class_text_features = {
        #     torch.tensor([idx]): text_features[idx] for idx in range(len(self.BASE_CLASSES_SPLIT1))
        # }
        self.visual_mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )
        self.final_linear = nn.Linear(512, 2048)
        self.fc = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU())

    def forward(self, vision_input):

        text_feature = self.text_features
        text_feature1 = text_feature.permute(1, 0).contiguous()

        q_visual = self.visual_mlp(vision_input)

        attention_scores = torch.matmul(q_visual, text_feature1)

        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_text_features = torch.matmul(attention_weights, text_feature)

        attended_features = self.final_linear(attended_text_features)

        combined_features = torch.cat([attended_features, vision_input], dim=1)

        output = self.fc(combined_features)

        return output