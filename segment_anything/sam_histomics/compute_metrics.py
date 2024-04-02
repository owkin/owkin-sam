"""Infer SAM on tiles from a DB and compute metrics."""

from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from histomics.data.datasets.from_database.nuclick_validation_dataset import (
    NuclickValidationFromDataBase,
)
from histomics.data.io.torch_dataset import TrainDataset
from histomics.data.transforms.label_transforms import (
    NuclickRandomCentroidLabelTransform,
)
from histomics.data.transforms.model_image_transforms import DummyTransform
from shapely import geometry

from segment_anything.sam_histomics.sam_inferer import SamInferer


def get_dataset_from_database(
    db_path: Optional[str] = None, sql_query: Optional[str] = None
) -> TrainDataset:
    """Get the ground truth DB tiles in a dataset format."""

    if db_path is None:
        db_path = "/home/owkin/project/database/ground_truth_tiles.db"
    root_path = f"sqlite:///{db_path}"

    # These tiles are the 6 ground truth tiles annotated by Katharina
    tiles_gt_db = "(3521541,6619042,65331257,84715806,96202074,67912873)"

    if sql_query is None:
        sql_query = (
            "SELECT tiles.tile_id,"
            " tiles.coords,tiles.slide_name,slides.abstra_slide_path,"
            " annotators.cytomine_username, tiles.timestamp_pushed,"
            " tiles.timestamp_row_creation, tiles.timestamp_pulled FROM slides"
            " INNER JOIN tiles ON tiles.slide_name=slides.slide_name"
            " INNER JOIN annotators ON tiles.annotator_id=annotators.annotator_id"
            " WHERE tiles.annotation_type IN ('_Priority_High','_Priority_Med','_Priority_Low')"
            f" AND tiles.is_pulled_from_cytomine=1 AND tiles.tile_id IN {tiles_gt_db}"
        )

    dataset_db = NuclickValidationFromDataBase(
        root_path=root_path, debug=False, sql_query=sql_query
    )

    label_transform = NuclickRandomCentroidLabelTransform(0.1)

    dataset = TrainDataset(
        dataset_db,
        label_transform=label_transform,
        model_transform=DummyTransform(),
    )

    return dataset


def get_inferer() -> SamInferer:
    """Load a pre-trained SAM model for inference."""

    sam_checkpoint = "sam_vit_b_01ec64.pth"
    inferer = SamInferer(
        model_type="vit_b",
        path_weights=f"/home/owkin/sam_weights/{sam_checkpoint}.pth",
        device=torch.device("cuda:0"),
    )

    return inferer


def transform_masks_to_contours(masks: np.ndarray) -> list:
    """Compute contours from boolean masks."""
    mask_binary = masks.astype(np.uint8)

    all_contours = []
    for mask in mask_binary:
        contours, _ = cv2.findContours(
            mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
        )

        if len(contours) > 1:
            all_contours.append(contours[0])

    return all_contours


def contour_list_to_dataframe(contours: list[np.ndarray]) -> pd.DataFrame:
    """Convert a list of contours into a DataFrame format.

    Parameters
    ----------
    contours : list[np.ndarray]
        A list of contours.

    Returns
    -------
    pd.DataFrame
        A DataFrame with a column "coordinates" containing the WKT representation
        of the polygons.
    """
    list_polygons = [geometry.Polygon(x).wkt for x in contours]

    df_polygon = pd.DataFrame({"coordinates": list_polygons})

    return df_polygon
