from .single_stage import SingleStageDetector
from mmdet.core import bbox_mask2result
from ..registry import DETECTORS


@DETECTORS.register_module
class FourierNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(FourierNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                         test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      gt_centers=None,
                      gt_max_centerness=None
                      ):

        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)

        losses = self.bbox_head.loss(
            *loss_inputs,
            gt_masks=gt_masks,
            gt_bboxes_ignore=gt_bboxes_ignore,
            gt_centers=gt_centers,
            gt_max_centerness=gt_max_centerness
        )
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

        results = [
            bbox_mask2result(det_bboxes, det_masks, det_labels, self.bbox_head.num_classes, img_meta[0])
            for det_bboxes, det_labels, det_masks in bbox_list]

        bbox_results = results[0][0]
        mask_results = results[0][1]

        return bbox_results, mask_results
