from ..classes import GuiderFrame
from typing import Optional, List
import numpy as np

from .clipping import get_clipped_mask_by_distance


def stack_frames(frames: List[GuiderFrame], centroids: np.ndarray, sigmaclip_val: Optional[float]) -> np.ndarray:
        """Returns the stacked guider frame data by averaging all frames
        after aligning them based on their fitted centroids.
        """
        mask = get_clipped_mask_by_distance(centroids, sigmaclip_val=sigmaclip_val)
        # Only stack non-outlier frames.
        # Use the centroid positions as offsets.
        ctr_used = centroids[mask]
        frames_used = [frames[i] for i in range(len(frames)) if mask[i]]
        # Determine the size of the stacked frame
        x_offsets = ctr_used[:, 0] - np.min(ctr_used[:, 0])
        y_offsets = ctr_used[:, 1] - np.min(ctr_used[:, 1])
        x_size = int(np.ceil(np.max(x_offsets))) + frames_used[0].data.shape[1]
        y_size = int(np.ceil(np.max(y_offsets))) + frames_used[0].data.shape[0]
        stacked_data = np.zeros((y_size, x_size), dtype=float)
        count_data = np.zeros((y_size, x_size), dtype=int)
        for frame, (x_off, y_off) in zip(frames_used, zip(x_offsets, y_offsets)):
            framedata = frame.data / frame.exptime
            frame_x, frame_y = frame.data.shape[1], frame.data.shape[0]
            x_off_int = int(np.round(x_off))
            y_off_int = int(np.round(y_off))
            stacked_data[
                y_off_int : y_off_int + frame_y,
                x_off_int : x_off_int + frame_x,
            ] += framedata
            count_data[
                y_off_int : y_off_int + frame_y,
                x_off_int : x_off_int + frame_x,
            ] += 1
        stacked_data /= np.maximum(count_data, 1)  # Avoid division by zero
        return stacked_data