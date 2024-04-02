"""Infer SAM-like models on histo tiles."""

from pathlib import Path
from typing import Any, Literal, Union

import numpy as np
import torch
from histomics.data.io.torch_dataset import TrainDataset
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from segment_anything import sam_model_registry
from segment_anything.predictor import SamPredictor


class SamInferer:
    """Class to infer a base SAM model on histological tiles."""

    def __init__(
        self,
        model_type: Literal["default", "vit_s", "vit_b", "vit_h"],
        path_weights: Union[Path, str],
        device: torch.device,
    ):
        self.model_type = model_type
        self.path_weights = path_weights
        self.device = device

        self.predictor = self._load_model()

    def _load_model(self) -> SamPredictor:
        """Load a model from a given path."""

        sam = sam_model_registry[self.model_type](checkpoint=self.path_weights)
        sam.to(device=self.device)

        predictor = SamPredictor(sam)
        logger.success(f"Successfully loaded weights from {self.path_weights}")

        return predictor

    @torch.no_grad()
    def infer_on_tile(self, image: np.ndarray, points: np.ndarray) -> list[np.ndarray]:
        """Infer on a tile.

        Parameters
        ----------
        image : np.ndarray
            The image to infer on. SHAPE (H, W, C)
        points : np.ndarray
            The points to infer on. SHAPE (N_POINTS, 2)

        Returns
        -------
        list[np.ndarray]
            The masks inferred for each point. Each mask is a np.ndarray of SHAPE (H, W)
        """

        self.predictor.set_image(image)

        point_coords = np.squeeze(points)
        nb_points = point_coords.shape[0]
        all_masks = []

        for i in tqdm(range(nb_points)):
            input_label = np.array([1])
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords[[i], :],
                point_labels=input_label,
                multimask_output=True,
            )
            # Select the best mask
            mask_max_score = masks[np.argmax(scores)]

        if (
            mask_max_score.astype(int).sum()
            > mask_max_score.shape[0] * mask_max_score.shape[1] * 0.3
        ):
            logger.warning(f"Mask is too large for point {i}, skipping.")
        else:
            all_masks.append(mask_max_score)

        return all_masks

    @torch.no_grad()
    def batch_inference(self, batch: list) -> tuple[np.ndarray, np.ndarray]:
        """Perform batch inference of all points on a single input tile.

        This method differs from the first one since it uses the predict_torch method
        of the SamPredictor, which is faster from the previous one, but requires
        pre-processing before fed to the model.

        Parameters
        ----------
        batch : list
            A batch coming from the `TrainDataset` class passed to a DataLoader.
            This function expects a batch size of one.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The masks and scores for each point in the batch.
        """
        # Go back to numpy for set_image
        img_numpy = np.squeeze(batch[0].numpy()).transpose(1, 2, 0)  # (H, W, C)

        self.predictor.set_image(img_numpy)

        point_coords = batch[1]["centroids"].permute(((1, 0, 2)))  # (N_POINTS, 1, 2)

        transformed_coords = self.predictor.transform.apply_coords_torch(
            point_coords, original_size=img_numpy.shape[:2]
        )

        transformed_coords = transformed_coords.to(self.device)

        labels = torch.ones(
            transformed_coords.shape[0], 1, device=self.device
        )  # (B, N_POINTS), here (N_POINTS, 1)

        masks, scores, _ = self.predictor.predict_torch(
            point_coords=transformed_coords,
            point_labels=labels,
            boxes=None,
            multimask_output=True,
        )

        masks = masks.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()

        return masks, scores

    def infer_on_dataset(
        self, dataset: TrainDataset
    ) -> dict[int, np.ndarray]:
        """Infer on an input TrainDataset.

        Parameters
        ----------
        dataset : TrainDataset
            The dataset to infer on.

        Returns
        -------
        dict[int, tuple[np.ndarray]]
            A dictionary containing the masks and the scores inferred for each point
            in the dataset. The key corresponds to the tile_id.
        """

        # Instantiate DataLoader
        dataloader = DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=1,
            collate_fn=None,
            drop_last=False,
        )

        outputs_all_dataset: dict[int, Any] = {}
        for batch in dataloader:
            tile_id = int(batch[2][0])
            masks, scores = self.batch_inference(batch)

            final_masks = self._post_process_mask(masks, scores)

            outputs_all_dataset[tile_id] = final_masks

        return outputs_all_dataset

    def _post_process_mask(self, masks: np.ndarray, scores: np.ndarray) -> torch.Tensor:
        """Post Process masks after SAM inference.

        1. Select the mask with the highest score
        2. Remove masks which take more than 30% of the image.

        Parameters
        ----------
        masks: np.ndarray
            The masks inferred by the model. SHAPE (N_POINTS, N_PREDICTIONS, H, W)
        scores: np.ndarray
            The scores associated with each mask. SHAPE (B,)

        Returns
        -------
        torch.Tensor
            The final mask. SHAPE (H, W)
        """

        processed_masks = np.zeros(
            (masks.shape[0], masks.shape[2], masks.shape[3]), dtype=bool
        )

        for idx_mask, mask in enumerate(masks):
            best_mask = mask[np.argmax(scores[idx_mask, :]), :, :]
            if (
                best_mask.astype(int).sum()
                > best_mask.shape[0] * best_mask.shape[1] * 0.3
            ):
                logger.warning(f"Mask is too large for point {idx_mask}, skipping.")
                continue

            processed_masks[idx_mask, :, :] = best_mask

        # Select the best mask

        return processed_masks
