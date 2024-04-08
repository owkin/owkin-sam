"""Infer SAM on tiles from a DB and compute metrics."""

import ast
import hashlib
from typing import Any, Optional

import cv2
import numpy as np
import openslide
import pandas as pd
import torch
from loguru import logger
from shapely import geometry, get_coordinates
from shapely.wkt import loads
from torch.utils.data import DataLoader

from histomics.data.datasets.from_database.nuclick_validation_dataset import (
    NuclickValidationFromDataBase,
)
from histomics.data.io.torch_dataset import TrainDataset
from histomics.data.transforms.label_transforms import (
    NuclickRandomCentroidLabelTransform,
)
from histomics.data.transforms.model_image_transforms import DummyTransform
from segment_anything.sam_histomics.sam_inferer import SamInferer


def get_dataset_from_database(
    db_path: Optional[str] = None, sql_query: Optional[str] = None
) -> TrainDataset:
    """Get the ground truth DB tiles in a dataset format."""

    if db_path is None:
        db_path = "/home/owkin/project/database/ground_truth_tiles.db"
    root_path = f"sqlite:///{db_path}"

    if sql_query is None:
        sql_query = (
            "SELECT tiles.tile_id, tiles.location_id,"
            " tiles.coords, tiles.slide_name, slides.abstra_slide_path,"
            " annotators.cytomine_username, tiles.timestamp_pushed,"
            " tiles.timestamp_row_creation, tiles.timestamp_pulled FROM slides"
            " INNER JOIN tiles ON tiles.slide_name=slides.slide_name"
            " INNER JOIN annotators ON tiles.annotator_id=annotators.annotator_id"
            " WHERE tiles.annotation_type IN ('_Priority_High','_Priority_Med','_Priority_Low')"
            " AND tiles.is_pulled_from_cytomine=1 AND annotators.cytomine_username='kvonloga'"
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
        path_weights=f"/home/owkin/sam_weights/{sam_checkpoint}",
        device=torch.device("cuda:0"),
    )

    return inferer


def export_masks_to_dataframe_annotations(
    dataset_outputs: dict[int, np.ndarray], dataset: TrainDataset
) -> pd.DataFrame:
    """Export outputs of a dataset to DB-like annotations in the Database.

    Parameters
    ----------
    dataset_outputs : dict[int, np.ndarray]
        The output of the model.
    dataset : TrainDataset
        The dataset from which the outputs come from.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the annotations.
    """
    all_dataframes = []

    df = dataset.dataset.dataframe

    for roi_name, mask in dataset_outputs.items():
        series = df.loc[df.index == roi_name].iloc[0]

        location_id = series["location_id"]

        contours = transform_masks_to_contours(mask)  # List of all contours

        # Coords of the tile
        coords_tile = ast.literal_eval(str(series["coords"]))  # (coords, level, size)

        # Open the slide in openslide to fetch the dimensions
        slide = openslide.open_slide(series["abstra_slide_path"])

        _, height = slide.dimensions

        # Convert the contours to the Cytomine format (reference in bottom left corner)
        contours_cytomine = [
            contour_to_cytomine_frame(contour, coords_tile, height)
            for contour in contours
        ]

        # Convert the contours to a DataFrame
        df_contours = contour_list_to_dataframe(contours_cytomine, location_id)

        df_contours["coords"] = series["coords"]
        df_contours["slide_name"] = series["slide_name"]
        df_contours["abstra_slide_path"] = series["abstra_slide_path"]
        df_contours["location_id"] = int(location_id)
        df_contours["cell_type"] = "Cell with unknown type"
        df_contours["cytomine_username"] = "saminference"

        all_dataframes.append(df_contours)

    return pd.concat(all_dataframes, axis=0).reset_index(drop=True)


def contour_to_cytomine_frame(
    contour: np.ndarray,
    coords_tile: tuple[tuple[int, int], int, tuple[int, int]],
    slide_height: int,
):
    """Transform a contour with "tile" frame coords to Cytomine frame coords.

    Parameters
    ----------
    contour : np.ndarray
        The contour to transform, shape (N_POINTS, 1, 2)
    coords_tile : tuple[tuple[int, int], int, tuple[int, int]]
        The coordinates of the tile in the format (coords, level, size).
    slide_height : tuple[int, int]
        The height of the slide (required to )

    Returns
    -------
    np.ndarray
        The transformed contour.
    """
    x_offset, y_offset = coords_tile[0]

    contour[:, 0, 0] = contour[:, 0, 0] + x_offset
    contour[:, 0, 1] = slide_height - (contour[:, 0, 1] + y_offset)

    return contour


def transform_masks_to_contours(masks: np.ndarray) -> list[np.ndarray]:
    """Compute contours from boolean masks.

    Parameters
    ----------
    masks : np.ndarray
        An array of boolean masks for the same image, shape (N_MASKS, H, W)

    Returns
    -------
    list[np.ndarray]
        A list of contours, of length N_MASKS.
    """
    mask_binary = masks.astype(np.uint8)

    all_contours = []
    for idx_mask, mask in enumerate(mask_binary):
        contours, _ = cv2.findContours(
            mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
        )
        if len(contours) == 0:
            logger.warning(f"Did not find any contour for mask {idx_mask}")
            continue
        elif len(contours) > 1:
            # If we find multiple contours, select the one with the max. area
            all_areas = [cv2.contourArea(contour) for contour in contours]
            all_contours.append(contours[np.argmax(all_areas)])
        else:
            all_contours.append(contours[0])

    return all_contours


def contour_list_to_dataframe(
    contours: list[np.ndarray], location_id: int
) -> pd.DataFrame:
    """Convert a list of contours into a DataFrame format.

    Parameters
    ----------
    contours : list[np.ndarray]
        A list of contours.

    location_id: int
        Location ID of the tile, required to create a fake cell id.

    Returns
    -------
    pd.DataFrame
        A DataFrame with a column "coordinates" containing the WKT representation
        of the polygons.
    """
    # TODO(ghorent): Convert to
    list_polygons = [geometry.Polygon(np.squeeze(x)).wkt for x in contours]

    list_cell_ids = [
        int(
            hashlib.sha1(
                (x + str(location_id) + "saminference").encode("utf-8")
            ).hexdigest(),
            16,
        )
        % (10**8)
        for x in list_polygons
    ]

    df_polygon = pd.DataFrame({"coordinates": list_polygons, "cell_id": list_cell_ids})

    return df_polygon


def get_ground_truth_from_batch(
    targets: Optional[list[dict[str, torch.Tensor]]],
) -> list[dict[str, np.ndarray]]:
    """Get the ground truth from a batch of images and targets.

    This function assumes that everything is already on the same device.

    Parameters
    ----------
    images: Union[Tensor,list[Tensor]]
        The images
    targets: Optional[list[dict[str,Tensor]]]
        The targets
    tile_names: Optional[list[str]]
        The tile names

    Returns
    -------
    list[dict[str, np.ndarray]]
        The ground_truth, in the format to fit in the
        metrics evaluation of DensePredictionMetric.
    """
    ground_truth: list[dict[str, np.ndarray]] = []
    assert isinstance(
        targets, list
    ), "Targets must be a list, when using ground truth to compute metrics"

    for target in targets:
        instance_ground_truth = {
            key: np.squeeze(value.detach().cpu().numpy())
            for key, value in target.items()
        }
        ground_truth.append(instance_ground_truth)
    return ground_truth


def aggregate_valid_metrics(
    metrics: dict[str, Any], log: bool = False
) -> dict[str, float]:
    """Aggregate validation metrics.

    Parameters
    ----------
    log: bool
        If need to log valid metric

    Returns
    -------
    dict[str, float]
        A dictionary containing the validation metrics
    """
    valid_metrics: dict[str, float] = {}
    for key in metrics:
        valid_metric = metrics[key].aggregate()
        # In the case where the metric computes multiple metrics
        if isinstance(valid_metric, dict):
            valid_metrics.update(valid_metric)
        else:
            valid_metrics[key] = valid_metric
        if log:
            logger.info(f"Valid {key}: {valid_metric}")
    return valid_metrics


def compute_metrics_on_outputs(
    outputs: dict[int, np.ndarray],
    dataset: TrainDataset,
    metrics: dict[str, Any],
    tile_by_tile: bool = True,
) -> pd.DataFrame:
    """Compute ordinary metrics on dataset outputs.

    Parameters
    ----------
    outputs : dict[int, np.ndarray]
        The outputs of the model, key: tile_id, value: boolean mask of
        shape (N_PRED, H, W)
    dataset : TrainDataset
        The dataset from which the outputs come from.
    metrics : dict[str, Any]
        The metrics to compute.
    tile_by_tile : bool
        Whether to compute the metrics tile by tile or on the whole dataset.
    """

    # Instantiate DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=None,
        drop_last=False,
    )

    scores_df = pd.DataFrame()

    for batch in dataloader:
        img, target, ids = batch
        tile_id = ids[0]

        # target is a list of
        ground_truth = get_ground_truth_from_batch([target])

        logger.info(
            f"Number of ground truth instances: {ground_truth[0]['masks'].shape[0]}"
        )
        logger.info(
            f"Number of prediction instances: {outputs[int(tile_id)].shape[0]}"
        )

        # Get the output from this tile
        predictions = [{"masks": outputs[int(tile_id)].astype(np.uint8)}]

        for metric_name, metric in metrics.items():
            assert (
                predictions is not None
            ), "Predictions should not be None if metrics are to be computed"

            metric.accumulate_sample_wise_metric(predictions, ground_truth)

        if tile_by_tile:
            tile_metrics = {}
            for metric_name, metric in metrics.items():
                agg_metric = metric.aggregate()
                if isinstance(agg_metric, dict):
                    tile_metrics.update(agg_metric)
                else:
                    tile_metrics[metric_name] = agg_metric

            scores = pd.DataFrame({tile_id: pd.Series(tile_metrics)}).T
            scores_df = pd.concat([scores_df, scores])

            for metric in metrics.values():
                metric.reset()

    if tile_by_tile:
        return scores_df

    valid_metrics = aggregate_valid_metrics(metrics, log=True)
    scores_df = pd.DataFrame(
        {
            "cohort": "Ground Truth tiles",
            "n_tiles": len(dataset),
            **valid_metrics,
        },
        index=[0],
    )

    return scores_df


def convert_contour_from_cytomine_to_tile_frame(
    contour: np.ndarray, slide_height: int, tile_coords: tuple[int, int]
):
    """Convert contours in Numpy array form from Cytomine frame to tile frame.

    Parameters
    ----------
    contour : np.ndarray
        The contour to convert, shape (N_POINTS, 2), in the Cytomine reference frame.
    slide_height : int
        The height of the slide.
    tile_coords : tuple[int, int]
        The coordinates of the tile.
    """
    # First convert coordinates from Cytomine to Owkin reference frame
    contour[:, 1] = (slide_height - contour[:, 1]) - tile_coords[1]
    contour[:, 0] = contour[:, 0] - tile_coords[0]

    return contour


def contours_to_masks(
    contours: list[geometry.Polygon],
    tile_coords: tuple[int, int],
    slide_height: int,
    img_shape: tuple[int, int],
) -> np.ndarray:
    """Convert a list of contours in Polygon format to a boolean mask.

    The goal is to convert annnotations from the DB to boolean masks usable
    to compute metrics.

    Parameters
    ----------
    contours : list[geometry.Polygon]
        The contours to convert.
    tile_coords : tuple[int, int]
        The coordinates of the tile.
    slide_height : int
        The height of the slide, to convert to the Owkin reference frame.
    img_shape : tuple[int, int]
        The shape of the image.

    Returns
    -------
    np.ndarray
        The boolean masks, shape (N_CONTOURS, H, W)
    """
    masks = np.zeros((len(contours), *img_shape), dtype=np.uint8)
    for idx_contour, contour in enumerate(contours):
        contour_coords = get_coordinates(contour)
        tile_contour_coords = convert_contour_from_cytomine_to_tile_frame(
            contour_coords, slide_height, tile_coords
        )

        cv2.fillPoly(
            masks[idx_contour, :, :], [tile_contour_coords.astype(np.int32)], color=1
        )
    return masks


def convert_df_annotations_to_masks(df_annotations: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame annotations to masks easily usable for metrics.

    Parameters
    ----------
    df_annotations : pd.DataFrame
        The DataFrame containing the annotations. This dataframe should contain the
        following columns:
        - tile_id: the tile id, should be unique (as we are working on one tile)
        - abstra_slide_path: the path to the slide
        - coords: the coordinates of the tile
        - coordinates: the coordinates of the polygons in WKT format

    Returns
    -------
    np.ndarray
        The masks, shape (N_MASKS, H, W)
    """

    assert df_annotations["tile_id"].nunique() == 1, "Only one tile_id is allowed"

    slide_path = df_annotations["abstra_slide_path"].iloc[0]
    slide = openslide.open_slide(slide_path)

    _, height = slide.dimensions
    list_polygons = df_annotations["coordinates"].apply(loads).tolist()

    full_tile_coords = ast.literal_eval(df_annotations["coords"].iloc[0])
    tile_coords = full_tile_coords[0]
    img_shape = full_tile_coords[2]

    masks = contours_to_masks(list_polygons, tile_coords, height, img_shape)

    return masks
