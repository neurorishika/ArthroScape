# arthroscape/sim/odor_sources.py
"""
External odor source module for ArthroScape.

This module provides classes for loading odor landscapes from external sources
such as images and videos. These can be used as static backgrounds or dynamic,
time-varying odor fields that agents respond to during simulation.

Classes:
    ImageOdorSource: Loads a static odor map from an image file.
    VideoOdorSource: Streams frames from a video as a dynamic odor field.

Example usage:
    # Static image as odor landscape
    from arthroscape.sim.odor_sources import ImageOdorSource
    source = ImageOdorSource("odor_map.png")
    source.apply_to_arena(arena)

    # Dynamic video as odor field
    from arthroscape.sim.odor_sources import VideoOdorSource
    source = VideoOdorSource("odor_video.mp4", arena)
    # In your simulation loop: source.update(arena, frame_index)
"""

import numpy as np
import logging
from typing import Optional, Tuple, Union, Callable, TYPE_CHECKING
from abc import ABC, abstractmethod
from pathlib import Path

if TYPE_CHECKING:
    from arthroscape.sim.arena import Arena

logger = logging.getLogger(__name__)


class OdorSource(ABC):
    """
    Abstract base class for external odor sources.

    An odor source provides a 2D array of odor concentrations that can be
    applied to an arena's odor grid.
    """

    @abstractmethod
    def get_odor_map(self, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Get the odor map resized to the target shape.

        Args:
            target_shape (Tuple[int, int]): Target shape (ny, nx) matching the arena grid.

        Returns:
            np.ndarray: 2D array of odor concentrations.
        """
        pass

    def apply_to_arena(
        self,
        arena: "Arena",
        mode: str = "replace",
        scale: float = 1.0,
        offset: float = 0.0,
    ) -> None:
        """
        Apply this odor source to an arena's odor grid.

        Args:
            arena (Arena): The arena object with an `odor_grid` attribute.
            mode (str): How to apply the odor map:
                - "replace": Replace the arena's odor grid entirely.
                - "add": Add to the existing odor grid.
                - "multiply": Multiply with the existing odor grid.
                - "max": Take the element-wise maximum.
            scale (float): Scaling factor applied to the odor values.
            offset (float): Offset added after scaling.
        """
        odor_map = self.get_odor_map((arena.ny, arena.nx))
        scaled_map = odor_map * scale + offset

        if mode == "replace":
            arena.odor_grid = scaled_map.copy()
        elif mode == "add":
            arena.odor_grid += scaled_map
        elif mode == "multiply":
            arena.odor_grid *= scaled_map
        elif mode == "max":
            arena.odor_grid = np.maximum(arena.odor_grid, scaled_map)
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'replace', 'add', 'multiply', or 'max'."
            )


class ImageOdorSource(OdorSource):
    """
    Load a static odor landscape from an image file.

    The image is converted to grayscale and normalized to [0, 1] by default.
    You can provide a custom normalization function for more control.

    Attributes:
        image_path (str): Path to the image file.
        raw_data (np.ndarray): The loaded image data (grayscale, float64).
        invert (bool): If True, inverts the image (1 - value).
        normalize_func (Callable): Custom normalization function.

    Example:
        >>> source = ImageOdorSource("gradient.png", invert=False)
        >>> source.apply_to_arena(arena, mode="replace", scale=10.0)
    """

    def __init__(
        self,
        image_path: Union[str, Path],
        invert: bool = False,
        normalize_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        channel: Optional[int] = None,
    ):
        """
        Initialize the ImageOdorSource.

        Args:
            image_path (Union[str, Path]): Path to the image file.
            invert (bool): If True, invert the image values (1 - normalized).
            normalize_func (Optional[Callable]): Custom function to normalize the image.
                Takes a numpy array and returns a normalized array.
                If None, uses min-max normalization to [0, 1].
            channel (Optional[int]): If provided, use only this channel (0=R, 1=G, 2=B).
                If None, converts to grayscale using standard luminance weights.
        """
        self.image_path = Path(image_path)
        self.invert = invert
        self.normalize_func = normalize_func
        self.channel = channel
        self._raw_data: Optional[np.ndarray] = None
        self._load_image()

    def _load_image(self) -> None:
        """Load and preprocess the image."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Pillow is required for ImageOdorSource. "
                "Install it with: pip install Pillow"
            )

        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        img = Image.open(self.image_path)

        # Handle channel selection or grayscale conversion
        if self.channel is not None:
            img_array = np.array(img)
            if len(img_array.shape) < 3:
                logger.warning(
                    f"Image is already grayscale, ignoring channel={self.channel}"
                )
                gray = img_array.astype(np.float64)
            else:
                gray = img_array[:, :, self.channel].astype(np.float64)
        else:
            gray = np.array(img.convert("L"), dtype=np.float64)

        # Normalize
        if self.normalize_func is not None:
            normalized = self.normalize_func(gray)
        else:
            # Default: min-max normalization to [0, 1]
            min_val, max_val = gray.min(), gray.max()
            if max_val > min_val:
                normalized = (gray - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(gray)

        # Invert if requested
        if self.invert:
            normalized = 1.0 - normalized

        self._raw_data = normalized
        logger.info(
            f"Loaded image '{self.image_path.name}' with shape {self._raw_data.shape}"
        )

    def get_odor_map(self, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Get the odor map resized to match the arena grid.

        Args:
            target_shape (Tuple[int, int]): Target shape (ny, nx).

        Returns:
            np.ndarray: Resized odor map.
        """
        from PIL import Image

        # Resize using PIL for high-quality interpolation
        img = Image.fromarray((self._raw_data * 255).astype(np.uint8))
        # Note: PIL resize takes (width, height), but our shape is (ny, nx) = (height, width)
        resized = img.resize(
            (target_shape[1], target_shape[0]), Image.Resampling.BILINEAR
        )
        return np.array(resized, dtype=np.float64) / 255.0

    @property
    def raw_data(self) -> np.ndarray:
        """Return the raw loaded image data."""
        return self._raw_data

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the original image shape (height, width)."""
        return self._raw_data.shape


class VideoOdorSource(OdorSource):
    """
    Stream video frames as a dynamic, time-varying odor field.

    Each frame of the video becomes the odor landscape for that time step.
    The video can be synchronized with simulation frames using various strategies.

    Attributes:
        video_path (str): Path to the video file.
        fps (float): Frame rate of the video.
        total_frames (int): Total number of frames in the video.
        current_frame (int): Current frame index.

    Example:
        >>> source = VideoOdorSource("plume.mp4")
        >>> source.apply_to_arena(arena)  # Apply first frame
        >>> # In simulation loop:
        >>> source.advance_frame()
        >>> source.apply_to_arena(arena)
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        loop: bool = True,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        frame_step: int = 1,
        invert: bool = False,
        normalize_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        channel: Optional[int] = None,
        preload: bool = False,
    ):
        """
        Initialize the VideoOdorSource.

        Args:
            video_path (Union[str, Path]): Path to the video file.
            loop (bool): If True, loop back to start when video ends.
            start_frame (int): First frame to use (0-indexed).
            end_frame (Optional[int]): Last frame to use (exclusive). None means use all.
            frame_step (int): Step between frames (e.g., 2 means use every other frame).
            invert (bool): If True, invert the frame values.
            normalize_func (Optional[Callable]): Custom normalization function.
            channel (Optional[int]): If provided, use only this channel (0=R, 1=G, 2=B).
            preload (bool): If True, preload all frames into memory (faster but uses more RAM).
        """
        self.video_path = Path(video_path)
        self.loop = loop
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_step = frame_step
        self.invert = invert
        self.normalize_func = normalize_func
        self.channel = channel
        self.preload = preload

        self._cap = None
        self._preloaded_frames: Optional[np.ndarray] = None
        self._current_frame_idx = 0
        self._current_frame_data: Optional[np.ndarray] = None

        self._open_video()

        if preload:
            self._preload_all_frames()
        else:
            self._read_frame(0)

    def _open_video(self) -> None:
        """Open the video file and read metadata."""
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV is required for VideoOdorSource. "
                "Install it with: pip install opencv-python"
            )

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise IOError(f"Could not open video: {self.video_path}")

        self._total_video_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._video_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._video_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate effective range
        self._effective_end = (
            self.end_frame if self.end_frame is not None else self._total_video_frames
        )
        self._effective_end = min(self._effective_end, self._total_video_frames)
        self._effective_frames = list(
            range(self.start_frame, self._effective_end, self.frame_step)
        )

        logger.info(
            f"Opened video '{self.video_path.name}': "
            f"{self._video_width}x{self._video_height}, {self._video_fps} fps, "
            f"{len(self._effective_frames)} effective frames"
        )

    def _preload_all_frames(self) -> None:
        """Preload all frames into memory."""
        import cv2

        frames = []
        for frame_idx in self._effective_frames:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()
            if ret:
                processed = self._process_frame(frame)
                frames.append(processed)
            else:
                logger.warning(f"Could not read frame {frame_idx}")

        self._preloaded_frames = np.array(frames)
        self._current_frame_data = self._preloaded_frames[0]
        logger.info(f"Preloaded {len(frames)} frames into memory")

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a raw video frame into an odor map."""
        import cv2

        # Channel selection or grayscale conversion
        if self.channel is not None:
            if len(frame.shape) == 3:
                # OpenCV uses BGR order
                channel_map = {0: 2, 1: 1, 2: 0}  # R=2, G=1, B=0 in BGR
                gray = frame[:, :, channel_map.get(self.channel, self.channel)].astype(
                    np.float64
                )
            else:
                gray = frame.astype(np.float64)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Normalize
        if self.normalize_func is not None:
            normalized = self.normalize_func(gray)
        else:
            normalized = gray / 255.0

        # Invert if requested
        if self.invert:
            normalized = 1.0 - normalized

        return normalized

    def _read_frame(self, logical_index: int) -> bool:
        """Read a specific frame by logical index (after applying start/step)."""
        if self._preloaded_frames is not None:
            if 0 <= logical_index < len(self._preloaded_frames):
                self._current_frame_data = self._preloaded_frames[logical_index]
                self._current_frame_idx = logical_index
                return True
            return False

        import cv2

        if logical_index >= len(self._effective_frames):
            if self.loop:
                logical_index = logical_index % len(self._effective_frames)
            else:
                return False

        video_frame_idx = self._effective_frames[logical_index]
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ret, frame = self._cap.read()

        if ret:
            self._current_frame_data = self._process_frame(frame)
            self._current_frame_idx = logical_index
            return True
        return False

    def get_odor_map(self, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Get the current frame's odor map resized to target shape.

        Args:
            target_shape (Tuple[int, int]): Target shape (ny, nx).

        Returns:
            np.ndarray: Resized odor map from current frame.
        """
        import cv2

        if self._current_frame_data is None:
            return np.zeros(target_shape, dtype=np.float64)

        # Resize using OpenCV
        # cv2.resize takes (width, height), shape is (ny, nx) = (height, width)
        resized = cv2.resize(
            self._current_frame_data,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized

    def advance_frame(self, steps: int = 1) -> bool:
        """
        Advance to the next frame(s).

        Args:
            steps (int): Number of logical frames to advance.

        Returns:
            bool: True if successful, False if end of video (and not looping).
        """
        new_idx = self._current_frame_idx + steps
        if new_idx >= len(self._effective_frames):
            if self.loop:
                new_idx = new_idx % len(self._effective_frames)
            else:
                return False
        return self._read_frame(new_idx)

    def seek_frame(self, logical_index: int) -> bool:
        """
        Seek to a specific logical frame index.

        Args:
            logical_index (int): The frame index to seek to.

        Returns:
            bool: True if successful.
        """
        return self._read_frame(logical_index)

    def reset(self) -> None:
        """Reset to the first frame."""
        self._read_frame(0)

    @property
    def current_frame(self) -> int:
        """Return the current logical frame index."""
        return self._current_frame_idx

    @property
    def total_frames(self) -> int:
        """Return the total number of effective frames."""
        return len(self._effective_frames)

    @property
    def fps(self) -> float:
        """Return the video's frame rate."""
        return self._video_fps

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the original video frame shape (height, width)."""
        return (self._video_height, self._video_width)

    def __del__(self):
        """Release video capture on deletion."""
        if self._cap is not None:
            self._cap.release()

    def close(self) -> None:
        """Explicitly release video resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoOdorReleaseStrategy:
    """
    An odor release strategy that updates the arena with video frames.

    This class wraps a VideoOdorSource and provides a method compatible with
    the simulation loop's odor update pattern. It can be used alongside
    agent-based odor release.

    Example:
        >>> video_strategy = VideoOdorReleaseStrategy(
        ...     "plume.mp4",
        ...     arena,
        ...     sync_mode="simulation_fps",
        ...     simulation_fps=60
        ... )
        >>> # In simulation loop:
        >>> video_strategy.update(arena, simulation_frame_index)
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        arena: "Arena",
        mode: str = "replace",
        scale: float = 1.0,
        offset: float = 0.0,
        sync_mode: str = "one_to_one",
        simulation_fps: float = 60.0,
        **video_kwargs: Union[bool, int, float, str, None],
    ):
        """
        Initialize the video-based odor release strategy.

        Args:
            video_path (Union[str, Path]): Path to the video file.
            arena (Arena): The arena to apply odor to.
            mode (str): Application mode ('replace', 'add', 'multiply', 'max').
            scale (float): Scaling factor for odor values.
            offset (float): Offset to add to odor values.
            sync_mode (str): Frame synchronization mode:
                - "one_to_one": Each simulation frame = one video frame.
                - "video_fps": Use video's native FPS, interpolate if needed.
                - "simulation_fps": Map simulation time to video time.
            simulation_fps (float): Simulation frame rate (used for sync calculations).
            **video_kwargs (Union[bool, int, float, str, None]): Additional arguments passed to VideoOdorSource.
        """
        self.video_source = VideoOdorSource(video_path, **video_kwargs)
        self.mode = mode
        self.scale = scale
        self.offset = offset
        self.sync_mode = sync_mode
        self.simulation_fps = simulation_fps

        # Precompute sync parameters
        if sync_mode == "video_fps":
            self._frame_ratio = self.video_source.fps / simulation_fps
        elif sync_mode == "simulation_fps":
            self._frame_ratio = simulation_fps / self.video_source.fps
        else:
            self._frame_ratio = 1.0

        self._last_video_frame = -1

    def update(self, arena: "Arena", simulation_frame: int) -> None:
        """
        Update the arena's odor grid based on the current simulation frame.

        Args:
            arena (Arena): The arena to update.
            simulation_frame (int): Current simulation frame index.
        """
        # Calculate which video frame to use
        if self.sync_mode == "one_to_one":
            video_frame = simulation_frame
        elif self.sync_mode == "video_fps":
            # Simulation is faster/slower than video
            video_frame = int(simulation_frame * self._frame_ratio)
        else:  # simulation_fps
            # Map simulation time to video frame
            sim_time = simulation_frame / self.simulation_fps
            video_frame = int(sim_time * self.video_source.fps)

        # Only update if we've moved to a new video frame
        if video_frame != self._last_video_frame:
            self.video_source.seek_frame(video_frame)
            self.video_source.apply_to_arena(
                arena, mode=self.mode, scale=self.scale, offset=self.offset
            )
            self._last_video_frame = video_frame

    def reset(self) -> None:
        """Reset the video source to the beginning."""
        self.video_source.reset()
        self._last_video_frame = -1


def load_odor_from_image(
    image_path: Union[str, Path],
    arena: "Arena",
    scale: float = 1.0,
    invert: bool = False,
    mode: str = "replace",
) -> ImageOdorSource:
    """
    Convenience function to load an image as the arena's odor landscape.

    Args:
        image_path (Union[str, Path]): Path to the image file.
        arena (Arena): The arena to apply odor to.
        scale (float): Scaling factor for odor values.
        invert (bool): If True, invert the image values.
        mode (str): Application mode ('replace', 'add', 'multiply', 'max').

    Returns:
        ImageOdorSource: The created odor source (for further manipulation if needed).

    Example:
        >>> from arthroscape.sim.odor_sources import load_odor_from_image
        >>> source = load_odor_from_image("gradient.png", arena, scale=5.0)
    """
    source = ImageOdorSource(image_path, invert=invert)
    source.apply_to_arena(arena, mode=mode, scale=scale)
    return source


def create_gradient_odor_map(
    shape: Tuple[int, int],
    direction: str = "horizontal",
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> np.ndarray:
    """
    Create a simple gradient odor map.

    This is a utility function for testing and simple experiments.

    Args:
        shape (Tuple[int, int]): Shape of the map (ny, nx).
        direction (str): Gradient direction:
            - "horizontal": Left to right.
            - "vertical": Bottom to top.
            - "radial": Center outward.
            - "radial_inward": Edge inward to center.
        min_val (float): Minimum odor value.
        max_val (float): Maximum odor value.

    Returns:
        np.ndarray: The gradient odor map.
    """
    ny, nx = shape

    if direction == "horizontal":
        gradient = np.linspace(min_val, max_val, nx)
        return np.tile(gradient, (ny, 1))

    elif direction == "vertical":
        gradient = np.linspace(min_val, max_val, ny)
        return np.tile(gradient.reshape(-1, 1), (1, nx))

    elif direction in ("radial", "radial_inward"):
        y_coords = np.linspace(-1, 1, ny)
        x_coords = np.linspace(-1, 1, nx)
        X, Y = np.meshgrid(x_coords, y_coords)
        distance = np.sqrt(X**2 + Y**2)
        distance = distance / distance.max()  # Normalize to [0, 1]

        if direction == "radial_inward":
            distance = 1.0 - distance

        return min_val + (max_val - min_val) * distance

    else:
        raise ValueError(
            f"Unknown direction: {direction}. "
            "Use 'horizontal', 'vertical', 'radial', or 'radial_inward'."
        )
