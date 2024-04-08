"""Infer SAM on ground truth tiles,"""

from histomics.metrics.segmentation_metrics import EnsembleDiceHoVerNet, PanopticQuality
from histomics.metrics.segmentation_nuclick_metrics import (
    InstanceBasedMaskMetricByClass,
    IoUByInstance,
)
from segment_anything.sam_histomics.compute_metrics import (
    compute_metrics_on_outputs,
    get_dataset_from_database,
    get_inferer,
)

if __name__ == "__main__":
    inferer = get_inferer()

    dataset = get_dataset_from_database()

    outputs_dataset = inferer.infer_on_dataset(dataset)

    metrics = {
        "ensemble_dice": EnsembleDiceHoVerNet(),
        "IoU_by_class": InstanceBasedMaskMetricByClass(
            IoUByInstance,
            categories=[2],
        ),
        "Panoptic quality": PanopticQuality(),
    }

    df_metrics = compute_metrics_on_outputs(outputs_dataset, dataset, metrics)

    df_metrics.to_csv(
        "/home/owkin/project/experiments/sam/sam_gt_tiles_05_04.csv", index=True
    )
