"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py
"""
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher

import gc
import time


from .modeling.transformer_decoder.fcclip_transformer_decoder import MaskPooling, get_classification_logits, get_cosine_similarity

VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]


@META_ARCH_REGISTRY.register()
class FCCLIP(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        train_metadata,
        test_metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        # FC-CLIP
        geometric_ensemble_alpha: float,
        geometric_ensemble_beta: float,
        ensemble_on_valid_mask: bool,
        order_masks_by: str,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # FC-CLIP args
        self.mask_pooling = MaskPooling()
        self.geometric_ensemble_alpha = geometric_ensemble_alpha
        self.geometric_ensemble_beta = geometric_ensemble_beta
        self.ensemble_on_valid_mask = ensemble_on_valid_mask
        self.order_masks_by = order_masks_by

        self.train_text_classifier = None
        self.test_text_classifier = None
        self.void_embedding = nn.Embedding(1, backbone.dim_latent) # use this for void

        self.train_class_names = self.prepare_class_names_from_metadata(train_metadata)

        self.query_embeddings = None
        self.num_templates = None
        self.category_list = None
        self.category_overlapping_mask = None
        #self.m2f_threshold_tensor = None
        #self.c_threshold_tensor = None

        # print(self.train_class_names)

        # _, self.train_num_templates, self.train_class_names = self.prepare_class_names_from_metadata(train_metadata, train_metadata)
        # self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(test_metadata, train_metadata)


    def preencode_text(self, text_list):
        self.category_list = [[text] for text in text_list]
        self.query_embeddings, self.num_templates = self.get_text_classifier_from_list(self.category_list)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        category_overlapping_list = []

        for test_label in self.category_list:
            category_overlapping_list.append(not set([l for label in self.train_class_names for l in label]).isdisjoint(set(test_label)))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.category_overlapping_mask = torch.tensor(
            category_overlapping_list, device=device, dtype=torch.long
        )

        #self.m2f_threshold_tensor = torch.tensor([.15] * len(self.category_list), device=device, dtype=torch.float32)
        #self.c_threshold_tensor = torch.tensor([.2] * len(self.category_list), device=device, dtype=torch.float32)


    # def prepare_class_names_from_metadata(self, metadata, train_metadata):
    #     def split_labels(x):
    #         res = []
    #         for x_ in x:
    #             x_ = x_.replace(', ', ',')
    #             x_ = x_.split(',') # there can be multiple synonyms for single class
    #             res.append(x_)
    #         return res
    #     # get text classifier
    #     try:
    #         class_names = split_labels(metadata.stuff_classes) # it includes both thing and stuff
    #         train_class_names = split_labels(train_metadata.stuff_classes)
    #     except:
    #         # this could be for insseg, where only thing_classes are available
    #         class_names = split_labels(metadata.thing_classes)
    #         train_class_names = split_labels(train_metadata.thing_classes)
    #     train_class_names = {l for label in train_class_names for l in label}
    #     category_overlapping_list = []
    #     for test_class_names in class_names:
    #         is_overlapping = not set(train_class_names).isdisjoint(set(test_class_names))
    #         category_overlapping_list.append(is_overlapping)
    #     category_overlapping_mask = torch.tensor(
    #         category_overlapping_list, dtype=torch.long)
        
    #     def fill_all_templates_ensemble(x_=''):
    #         res = []
    #         for x in x_:
    #             for template in VILD_PROMPT:
    #                 res.append(template.format(x))
    #         return res, len(res) // len(VILD_PROMPT)
       
    #     num_templates = []
    #     templated_class_names = []
    #     for x in class_names:
    #         templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
    #         templated_class_names += templated_classes
    #         num_templates.append(templated_classes_num) # how many templates for current classes
    #     class_names = templated_class_names
    #     #print("text for classification:", class_names)
    #     return category_overlapping_mask, num_templates, class_names

    def prepare_class_names_from_metadata(self, metadata):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',') # there can be multiple synonyms for single class
                res.append(x_)
            return res
        # get text classifier
        try:
            class_names = split_labels(metadata.stuff_classes) # it includes both thing and stuff
        except:
            # this could be for insseg, where only thing_classes are available
            class_names = split_labels(metadata.thing_classes)

        print("TRAIN CLASS NAMES", class_names)
        
        #print("text for classification:", class_names)
        return class_names
    
    def prepare_class_names_from_list(self, query_list):
        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)
       
        num_templates = []
        templated_class_names = []
        for x in query_list:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num) # how many templates for current classes
        class_names = templated_class_names
        #print("text for classification:", class_names)
        return num_templates, class_names

    def set_metadata(self, metadata):
        self.test_metadata = metadata
        # self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(metadata, self.train_metadata)
        self.test_text_classifier = None
        return

    def get_text_classifier(self):
        if self.training:
            if self.train_text_classifier is None:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.train_class_names), bs):
                    text_classifier.append(self.backbone.get_text_classifier(self.train_class_names[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.train_text_classifier = text_classifier
            return self.train_text_classifier, self.train_num_templates
        else:
            if self.test_text_classifier is None:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.test_class_names), bs):
                    # print("TEST CLASS NAMES TYPE", type(self.test_class_names))
                    # print("TEST CLASS NAMES", self.test_class_names)
                    
                    text_classifier.append(self.backbone.get_text_classifier(self.test_class_names[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.test_text_classifier = text_classifier
            return self.test_text_classifier, self.test_num_templates
        
    def get_text_classifier_from_list(self, word_list):
        text_classifier = []
        # this is needed to avoid oom, which may happen when num of class is large
        bs = 128
        num_templates, word_list = self.prepare_class_names_from_list(word_list)
        # for i in range(0, len(word_list), len(VILD_PROMPT)):
        #     print(word_list[i: i + len(VILD_PROMPT)])
        for idx in range(0, len(word_list), bs):
            text_classifier.append(self.backbone.get_text_classifier(word_list[idx:idx+bs], self.device).detach())
        text_classifier = torch.cat(text_classifier, dim=0)

        # average across templates and normalization.
        text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
        text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
        text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
        self.test_text_classifier = text_classifier
        return self.test_text_classifier, num_templates

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "train_metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "test_metadata": MetadataCatalog.get(cfg.DATASETS.TEST[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "geometric_ensemble_alpha": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_ALPHA,
            "geometric_ensemble_beta": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_BETA,
            "ensemble_on_valid_mask": cfg.MODEL.FC_CLIP.ENSEMBLE_ON_VALID_MASK,
            "order_masks_by": cfg.MODEL.FC_CLIP.ORDER_MASKS_BY,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        # print(f"device is {self.device}")

        import time

        start = time.time()
        features = self.backbone(images.tensor)
        print(f"backbone done in {time.time() - start} s")

        start = time.time()

        text_classifier, num_templates = self.get_text_classifier()
        # Append void class weight
        text_classifier = torch.cat([text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0)
        features['text_classifier'] = text_classifier
        features['num_templates'] = num_templates

        print(f"text encoding done in {time.time() - start} s")

        start = time.time()
        outputs = self.sem_seg_head(features)
        print(f"sem seg head done in {time.time() - start} s")

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            print(mask_cls_results.shape, mask_pred_results.shape)

            # We ensemble the pred logits of in-vocab and out-vocab
            clip_feature = features["clip_vis_dense"]
            print(f"CLIP feature shape: {clip_feature.shape}")
            mask_for_pooling = F.interpolate(mask_pred_results, size=clip_feature.shape[-2:],
                                                mode='bilinear', align_corners=False)
            if "convnext" in self.backbone.model_name.lower():
                pooled_clip_feature = self.mask_pooling(clip_feature, mask_for_pooling)
                pooled_clip_feature = self.backbone.visual_prediction_forward(pooled_clip_feature)
            elif "rn" in self.backbone.model_name.lower():
                pooled_clip_feature = self.backbone.visual_prediction_forward(clip_feature, mask_for_pooling)
            else:
                raise NotImplementedError

            out_vocab_cls_results = get_classification_logits(pooled_clip_feature, text_classifier, self.backbone.clip_model.logit_scale, num_templates)
            in_vocab_cls_results = mask_cls_results[..., :-1] # remove void
            out_vocab_cls_results = out_vocab_cls_results[..., :-1] # remove void

            # Reference: https://github.com/NVlabs/ODISE/blob/main/odise/modeling/meta_arch/odise.py#L1506
            out_vocab_cls_probs = out_vocab_cls_results.softmax(-1)
            in_vocab_cls_results = in_vocab_cls_results.softmax(-1)
            category_overlapping_mask = self.category_overlapping_mask.to(self.device)

            if self.ensemble_on_valid_mask:
                # print("#" * 100)
                # print("ENSEMBLES ON VALID MASKS")
                # print("#" * 100)
                # Only include out_vocab cls results on masks with valid pixels
                # We empirically find that this is important to obtain reasonable AP/mIOU score with ResNet CLIP models
                valid_masking = (mask_for_pooling > 0).to(mask_for_pooling).sum(-1).sum(-1) > 0
                valid_masking = valid_masking.to(in_vocab_cls_results.dtype).unsqueeze(-1)
                alpha = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_alpha
                beta = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_beta
                alpha = alpha * valid_masking
                beta = beta * valid_masking
            else:
                alpha = self.geometric_ensemble_alpha
                beta = self.geometric_ensemble_beta

            cls_logits_seen = (
                (in_vocab_cls_results ** (1 - alpha) * out_vocab_cls_probs**alpha).log()
                * category_overlapping_mask
            )
            cls_logits_unseen = (
                (in_vocab_cls_results ** (1 - beta) * out_vocab_cls_probs**beta).log()
                * (1 - category_overlapping_mask)
            )
            cls_results = cls_logits_seen + cls_logits_unseen

            # This is used to filtering void predictions.
            is_void_prob = F.softmax(mask_cls_results, dim=-1)[..., -1:]
            mask_cls_probs = torch.cat([
                cls_results.softmax(-1) * (1.0 - is_void_prob),
                is_void_prob], dim=-1)
            mask_cls_results = torch.log(mask_cls_probs + 1e-8)

            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            processed_results[-1]["outputs"] = {}
            processed_results[-1]["outputs"]["pred_logits"] = mask_cls_result
            processed_results[-1]["outputs"]["pred_masks"] = mask_pred_result

            return processed_results
        
    def __order_masks_by_size(self, masks):
        masked_out_percentages = []

        for i in range(0, masks.shape[0]):

            masked_image = masks[i]

            masked_image = masked_image.sigmoid()

            masked_image = masked_image < 0.5

            masked_image = masked_image.squeeze().squeeze()

            # plt.imshow(masked_image.cpu())
            # plt.show()

            # pixels containing ones are masked out
            mskd_out_prcntg = torch.count_nonzero(masked_image) / masked_image.shape[0] * masked_image.shape[1]

            masked_out_percentages.append(mskd_out_prcntg)

        sort_index = torch.argsort(torch.tensor(masked_out_percentages))

        # sort_index = list(sort_index)

        # sort_index.reverse()

        sort_index.data = sort_index.flip(0)

        return sort_index
    
    def _order_masks_by_selection_score(self, mask_cls_results):

        sort_index = torch.argsort(mask_cls_results)

        return sort_index

    def get_image_embeddings(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        # start = time.time()
        features = self.backbone(images.tensor)
        # print(f"backbone done in {time.time() - start} s")

        # start = time.time()
        # text_classifier, num_templates = self.get_text_classifier_from_list([])
        # Append void class weight
        text_classifier = F.normalize(self.void_embedding.weight, dim=-1)
        features['text_classifier'] = text_classifier
        # features['num_templates'] = []

        # print(f"text encoding done in {time.time() - start} s")

        # start = time.time()
        outputs = self.sem_seg_head.get_image_embeddings(features)
        # print(f"sem seg head done in {time.time() - start} s")

        assert self.training == False
        # start = time.time()

        if self.order_masks_by == "size":
            filter_index = outputs["pred_logits"].squeeze() < 9.2
            # print(filter_index.shape, filter_index.sum())
            mask_pred_results = outputs["pred_masks"].squeeze()[filter_index].unsqueeze(0)
            mask2former_embeddings = outputs["class_embed"].squeeze()[filter_index].unsqueeze(0)

            # sort_index = self._order_masks_by_selection_score(outputs["pred_logits"].squeeze())[:30]

            # # update mask pred results by indexing with sort index
            # mask_pred_results = outputs["pred_masks"].squeeze()[sort_index].unsqueeze(0)

            # mask2former_embeddings = outputs["class_embed"].squeeze()[sort_index].unsqueeze(0)

            sort_index = self.__order_masks_by_size(mask_pred_results.squeeze())

            # print(sort_index.shape)

            mask_pred_results = mask_pred_results.squeeze()[sort_index].unsqueeze(0)
            mask2former_embeddings = mask2former_embeddings.squeeze()[sort_index]

            # print(mask_pred_results.shape, mask2former_embeddings.shape)
        elif self.order_masks_by == "confidence":

            # # filter outputs pred logits by threshold
            # filter_index = outputs["pred_logits"].squeeze() < 7
            # # TODO: manage the case in which there is only one mask, skip when there are no masks
            # if filter_index.sum() > 1:
            #     print(filter_index.shape, filter_index.sum())
            #     outputs["pred_logits"] = outputs["pred_logits"].squeeze()[filter_index].unsqueeze(0)
            #     outputs["pred_masks"] = outputs["pred_masks"].squeeze()[filter_index].unsqueeze(0)
            #     outputs["class_embed"] = outputs["class_embed"].squeeze()[filter_index].unsqueeze(0)

            sort_index = self._order_masks_by_selection_score(outputs["pred_logits"].squeeze())[:30]

            # update mask pred results by indexing with sort index
            mask_pred_results = outputs["pred_masks"].squeeze()[sort_index].unsqueeze(0)

            mask2former_embeddings = outputs["class_embed"].squeeze()[sort_index]

        elif self.order_masks_by == "hybrid":
            sort_index = self._order_masks_by_selection_score(outputs["pred_logits"].squeeze())[:30]

            # update mask pred results by indexing with sort index
            mask_pred_results = outputs["pred_masks"].squeeze()[sort_index].unsqueeze(0)
            mask2former_embeddings = outputs["class_embed"].squeeze()[sort_index]

            sort_index = self.__order_masks_by_size(mask_pred_results.squeeze())

            mask_pred_results = mask_pred_results.squeeze()[sort_index].unsqueeze(0)
            mask2former_embeddings = mask2former_embeddings.squeeze()[sort_index]

        else:
            raise NotImplementedError
        
        # print(f"mask ordering done in {time.time() - start} s")

        # del outputs
        # gc.collect()
        # torch.cuda.empty_cache()

        # start = time.time()

        # We ensemble the pred logits of in-vocab and out-vocab
        clip_feature = features["clip_vis_dense"]
        mask_for_pooling = F.interpolate(mask_pred_results, size=clip_feature.shape[-2:],
                                            mode='bilinear', align_corners=False)
        if "convnext" in self.backbone.model_name.lower():
            pooled_clip_feature = self.mask_pooling(clip_feature, mask_for_pooling)
            pooled_clip_feature = self.backbone.visual_prediction_forward(pooled_clip_feature)
        elif "rn" in self.backbone.model_name.lower():
            pooled_clip_feature = self.backbone.visual_prediction_forward(clip_feature, mask_for_pooling)
        else:
            raise NotImplementedError
        
        # print(f"pooling done in {time.time() - start} s")

        # start = time.time()
        
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )

        mask_pred_results = mask_pred_results.squeeze()

        image_size = images.image_sizes[0]

        input_image_height = batched_inputs[0]['height']
        input_image_width = batched_inputs[0]['width']

        if self.sem_seg_postprocess_before_inference:
                    mask_pred_results = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_results, image_size, input_image_height, input_image_width
                    )

        # print(f"postprocessing done in {time.time() - start} s")

        clip_embeddings = pooled_clip_feature.squeeze()

        # mask2former_embeds_size = mask2former_embeddings.shape[-1]
        # clip_embeddings_size = clip_embeddings.shape[-1]
        # instead of the embedding itself, save the index of the embedding from the embedding list
        index_image = torch.zeros(input_image_height, input_image_width, 1).cuda() - 1

        feat_shape = index_image.shape

        # start = time.time()

        double_feats = torch.cat((mask2former_embeddings, clip_embeddings), axis=-1)

        for idx, masked_image in enumerate(mask_pred_results):

            masked_image = masked_image.sigmoid_() < 0.5
            masked_image = masked_image.squeeze().unsqueeze(-1).expand(feat_shape)

            assert masked_image.shape[:2] == (input_image_height, input_image_width), masked_image.shape

            index_image[:] = torch.where((masked_image == 0) & (index_image == -1), idx, index_image)

        # print(f"Feature embedding done in {time.time() - start} s")

        return [(double_feats, index_image)]
    
    def query_image_segmentation(self, query_list, featured_voxel, index_image):

        max_indices = self.query_segmentation(query_list, featured_voxel)

        category_image = torch.zeros(index_image.shape).cuda() - 1

        for idx, max_index in enumerate(max_indices[0]):

            category_image[:] = torch.where((index_image == idx) & (category_image == -1), max_index, category_image)

        return category_image
    
    def get_relevancy_score(self, rendered_embeddings, query_embeddings):

        """
        Computes the relevancy score between rendered embeddings and query embeddings, compared to the canonical embeddings, as presented in https://arxiv.org/pdf/2303.09553
        Args:
            rendered_embeddings: Tensor of shape (N, D), where N is the number of rendered embeddings and D is the embedding dimension.
            query_embeddings: Tensor of shape (M, D), where M is the number of query embeddings and D is the embedding dimension.
        Returns:
            relevancy_scores: Tensor of shape (N, M) containing the relevancy scores.
        """
        pos_embeds = query_embeddings

        neg_embeds, _ = self.get_text_classifier_from_list([["background"], ["surroundings"], ["environment"], ["setting"], ["flat"], ["scenery"], ["scene"], ["landscape"]])

        device, dtype = rendered_embeddings.device, rendered_embeddings.dtype

        # concat pos + neg phrase embeddings
        phrases_embeds = torch.cat([pos_embeds, neg_embeds], dim=0).to(device=device, dtype=dtype)  
        output = torch.mm(rendered_embeddings, phrases_embeds.T)   # (rays, total_phrases)

        n_pos = pos_embeds.shape[0]  # number of positive phrases
        n_neg = neg_embeds.shape[0]  # number of negative phrases

        positive_vals = output[:, :n_pos]          # (rays, n_pos)
        negative_vals = output[:, n_pos:]          # (rays, n_neg)

        # expand positives to compare with all negatives
        repeated_pos = positive_vals.unsqueeze(2).expand(-1, -1, n_neg)   # (rays, n_pos, n_neg)
        negative_vals = negative_vals.unsqueeze(1).expand(-1, n_pos, -1)  # (rays, n_pos, n_neg)

        # stack into pairwise sims
        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # (rays, n_pos, n_neg, 2)

        # apply softmax over pos vs neg
        softmax = torch.softmax(10 * sims, dim=-1)  # (rays, n_pos, n_neg, 2)

        # find hardest negative for each positive
        best_id = softmax[..., 0].argmin(dim=2)  # (rays, n_pos)

        # gather the corresponding softmax row
        gathered = torch.gather(
            softmax, 
            2, 
            best_id.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)  # (rays, n_pos, 1, 2)
        ).squeeze(2)  # (rays, n_pos, 2)

        # we want only pos_prob (the prob that the positive wins)
        result = gathered[..., 0:1].permute((2, 0, 1))  # (rays, n_pos, 1)

        return result
    
    def query_segmentation(self, query_list, featured_voxel, m2f_cos_sim_threshold = 0.15, clip_cos_sim_threshold = 0.23, similarity_method='weighted_avg'): #weighted_avg_with_relevancy_score

        # word_list = prompt_labels(labels=category_list, prompt="photo")

        query_embeddings, num_templates = self.get_text_classifier_from_list(query_list)

        # Vectorized embeddings extraction (batch processing instead of pixel-by-pixel)
        # TODO: remove fixed size CLIP 
        mask2former_embeddings = featured_voxel[:, :, :768]
        clip_embeddings = featured_voxel[:, :, 768:]

        mask2former_embeddings = mask2former_embeddings.reshape(-1, mask2former_embeddings.shape[-1])
        clip_embeddings = clip_embeddings.reshape(-1, clip_embeddings.shape[-1])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        m2f_cosine_similarity = get_cosine_similarity(mask2former_embeddings, query_embeddings)
        c_cosine_similarity = get_cosine_similarity(clip_embeddings, query_embeddings)

        if similarity_method == 'weighted_avg_with_relevancy_score':

            m2f_relevancy_score = self.get_relevancy_score(mask2former_embeddings, query_embeddings)
            c_relevancy_score = self.get_relevancy_score(clip_embeddings, query_embeddings)

            m2f_relevancy_criterion = m2f_relevancy_score > 0.5
            c_relevancy_criterion = c_relevancy_score > 0.5

        # reshape back to original shape
        mask2former_embeddings = mask2former_embeddings.reshape(featured_voxel.shape[:-1] + (mask2former_embeddings.shape[-1],))
        clip_embeddings = clip_embeddings.reshape(featured_voxel.shape[:-1] + (clip_embeddings.shape[-1],))
        m2f_cosine_similarity = m2f_cosine_similarity.reshape(featured_voxel.shape[:-1] + (query_embeddings.shape[0], ))
        c_cosine_similarity = c_cosine_similarity.reshape(featured_voxel.shape[:-1] + (query_embeddings.shape[0], ))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        category_overlapping_list = []
        for test_label in query_list:
            category_overlapping_list.append(not set([l for label in self.train_class_names for l in label]).isdisjoint(set(test_label)))

        category_overlapping_mask = torch.tensor(
            category_overlapping_list, device=device, dtype=torch.int8
        )

        if similarity_method == 'weighted_avg_with_relevancy_score':
            relevancy_criterion = torch.where(
                category_overlapping_mask.bool().expand(m2f_relevancy_criterion.shape),
                m2f_relevancy_criterion,
                c_relevancy_criterion
            )

        indexed_similarity = torch.where(category_overlapping_mask.bool().expand(m2f_cosine_similarity.shape), m2f_cosine_similarity > m2f_cos_sim_threshold, c_cosine_similarity > clip_cos_sim_threshold)
        
        mask = indexed_similarity.max(dim=-1).values.bool()

        if similarity_method == 'weighted_avg':
            mask2former_logit_scale = self.sem_seg_head.predictor.logit_scale
            mask_cls_results = get_classification_logits(mask2former_embeddings, query_embeddings.unsqueeze(-1), mask2former_logit_scale, num_templates)
            out_vocab_cls_results = get_classification_logits(clip_embeddings, query_embeddings.unsqueeze(-1), self.backbone.clip_model.logit_scale, num_templates)

            in_vocab_cls_results = mask_cls_results

            # Reference: https://github.com/NVlabs/ODISE/blob/main/odise/modeling/meta_arch/odise.py#L1506
            out_vocab_cls_probs = out_vocab_cls_results.softmax(-1)
            in_vocab_cls_probs = in_vocab_cls_results.softmax(-1)

            category_overlapping_list = []
            for test_label in query_list:
                category_overlapping_list.append(not set([l for label in self.train_class_names for l in label]).isdisjoint(set(test_label)))

            category_overlapping_mask = torch.tensor(
                category_overlapping_list, device=device, dtype=torch.int8
            )

            alpha = self.geometric_ensemble_alpha
            beta = self.geometric_ensemble_beta
                
            cls_logits_seen = (
                (in_vocab_cls_probs ** (1 - alpha) * out_vocab_cls_probs**alpha).log()
                * category_overlapping_mask
            )
            cls_logits_unseen = (
                (in_vocab_cls_probs ** (1 - beta) * out_vocab_cls_probs**beta).log()
                * (1 - category_overlapping_mask)
            )
            cls_results = cls_logits_seen + cls_logits_unseen

            pred_open_probs = F.softmax(cls_results, dim=-1)

            max_scores, max_indices = pred_open_probs.detach().max(dim=-1)

            max_indices = torch.where(mask, max_indices, len(query_list) if ["other"] not in query_list else query_list.index(["other"]))
            
        elif similarity_method == 'weighted_avg_with_relevancy_score':


            mask2former_logit_scale = self.sem_seg_head.predictor.logit_scale
            mask_cls_results = get_classification_logits(mask2former_embeddings, query_embeddings.unsqueeze(-1), mask2former_logit_scale, num_templates)
            out_vocab_cls_results = get_classification_logits(clip_embeddings, query_embeddings.unsqueeze(-1), self.backbone.clip_model.logit_scale, num_templates)

            in_vocab_cls_results = mask_cls_results

            # Reference: https://github.com/NVlabs/ODISE/blob/main/odise/modeling/meta_arch/odise.py#L1506
            out_vocab_cls_probs = out_vocab_cls_results.softmax(-1)
            in_vocab_cls_probs = in_vocab_cls_results.softmax(-1)

            category_overlapping_list = []
            for test_label in query_list:
                category_overlapping_list.append(not set([l for label in self.train_class_names for l in label]).isdisjoint(set(test_label)))

            category_overlapping_mask = torch.tensor(
                category_overlapping_list, device=device, dtype=torch.int8
            )

            alpha = self.geometric_ensemble_alpha
            beta = self.geometric_ensemble_beta
                
            cls_logits_seen = (
                (in_vocab_cls_probs ** (1 - alpha) * out_vocab_cls_probs**alpha).log()
                * category_overlapping_mask
            )
            cls_logits_unseen = (
                (in_vocab_cls_probs ** (1 - beta) * out_vocab_cls_probs**beta).log()
                * (1 - category_overlapping_mask)
            )
            cls_results = cls_logits_seen + cls_logits_unseen

            # This is used to filtering void predictions.
            # is_void_prob = F.softmax(mask_cls_results, dim=-1)[..., -1:]
            # mask_cls_probs = torch.cat([
            #     cls_results.softmax(-1) * (1.0 - is_void_prob),
            #     is_void_prob], dim=-1)
            # mask_cls_results = torch.log(mask_cls_probs + 1e-8)

            pred_open_probs = F.softmax(cls_results, dim=-1)

            # max_scores, max_indices = pred_open_probs.detach().max(dim=-1)

            # max_indices = max_indices.unsqueeze(-1).expand(pred_open_probs.shape).shape

            mask = mask.unsqueeze(-1).expand(pred_open_probs.shape)

            max_scores, max_indices = torch.where(torch.logical_and(mask, relevancy_criterion), pred_open_probs, -1).max(dim=-1)

            max_indices = torch.where(torch.logical_and(max_scores == -1, max_indices == 0), len(query_list) if ["other"] not in query_list else query_list.index(["other"]), max_indices)

            # # add an extra column to the relevancy criterion settled to always true to account for the "other" category
            # relevancy_criterion = torch.cat([relevancy_criterion.squeeze(), torch.ones((relevancy_criterion.shape[1], 1), dtype=torch.bool, device=device)], dim=-1)
            # criterion_mask = torch.gather(relevancy_criterion, 1,  max_indices)
            
            # # replace the indices that don't respect the relevancy criterion with the index of the "other" category
            # max_indices = torch.where(criterion_mask, max_indices, len(query_list) if ["other"] not in query_list else query_list.index(["other"]))
        else:
            max_indexed_similarity = torch.where(category_overlapping_mask.bool().expand(m2f_cosine_similarity.shape)[:, :, 0], m2f_cosine_similarity.argmax(dim=-1), c_cosine_similarity.argmax(dim=-1))

            max_indices = torch.where(mask, max_indexed_similarity, len(query_list) if ["other"] not in query_list else query_list.index(["other"]))

        #max_values = torch.where(category_overlapping_mask.bool().expand(m2f_cosine_similarity.shape)[:, :, 0], m2f_cosine_similarity.max(dim=-1).values, c_cosine_similarity.max(dim=-1).values)
        #print(max_values.max())

        zero_embeddings_index = torch.abs(featured_voxel).sum(dim=-1) == 0
        max_indices[zero_embeddings_index] = len(query_list) + 1 if ["other"] not in query_list else len(query_list)

        return max_indices

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        num_classes = len(self.test_metadata.stuff_classes)
        keep = labels.ne(num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.test_metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # if this is panoptic segmentation
        if self.panoptic_on:
            num_classes = len(self.test_metadata.stuff_classes)
        else:
            num_classes = len(self.test_metadata.thing_classes)
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.test_metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
