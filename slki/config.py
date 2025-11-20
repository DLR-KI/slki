# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
from typing import Any, Literal


TImportType = Literal["hdf5"]
TChunking = Literal["never", "onload", "onsave"]
TSensor = Literal["frog", "points", "both"]
TSensorData = Literal["adxl373-sh", "adxl373-sm", "kionix-sh", "kionix-sm"]
TAxis = Literal["x", "y", "z", "a"]
TPrecision = Literal["float16", "float32", "float64"]
TNorm = Literal["mean_var", "zero_one", "mone_one", "mone_one_zero_fix", "l1", "l2", "max"]
TSmoothType = Literal["conv", "exp"]


class Config:
    """Configuration class for defining pipeline parameters and settings."""

    # #########################################
    # ######### PIPELINE DEFINITION ###########
    # #########################################

    # Define whether to show a tqdm progress bar
    TQDM_OPTIONS: dict[str, Any] = dict(
        disable=False,
        leave=False,
        ncols=80,
    )
    TQDM_MAX_WORKERS: int | None = None

    # Define the import sources.
    # This can be a single or a list of file paths or diretory paths which can be contain path patterns.
    IMPORT_SOURCES: str | list[str] = "/home/lab/slki/Dataset/InferenzDaten20250606"

    # Define the data type. Available options are:
    # hdf5  - import hdf5 files from IMPORT_DIR
    IMPORT_TYPE: TImportType = "hdf5"

    # Define the maximum number of samples per sensor type that should be imported.
    # If None, all samples are imported.
    MAX_SAMPLES_PER_SENDOR_TYPE: int | None = None

    # Define the chunk point. Available options are:
    # never  - do not chunk the data
    # onload - chunk the data on load (before preprocessing)
    # onsave - chunk the data on save (after preprocessing)
    CHUNK_POINT: TChunking = "onload"

    # Define the chunk size. If None, the data will not be chunked.
    CHUNK_SIZE: int | None = 1000

    # Define which sensor data is used. Available options are:
    # points - use data from the point sensor embedded within the track
    # frog   - use data from the frog sensor embedded within the track
    # both   - concatenate the data from both sensors (data must not be 50/50, due to problems w/ wakeups) point|frog
    SENSOR: TSensor = "points"

    # Define which sensor data should be used. Available options are:
    # adxl373-sh
    # adxl373-sm
    # kionix-sh
    # kionix-sm
    # NOTE: Markus said "kionix-s{h,m}" are the correct sensors!
    #       "-sh" is a bit more precise, but might wake up too late
    #       "-sm" does not have that much precision, but is "always" awake
    SENSOR_DATA: TSensorData = "kionix-sh"

    # Define which axis should be used. Available options are:
    # x - use the first axis
    # y - use the second axis
    # z - use the third axis
    # a - average all the axis into one value (mean along the first axis)
    AXIS: TAxis = "z"

    # Define the precision of the data. Available options are:
    # float16
    # float32
    # float64
    PRECISION: TPrecision = "float32"

    # Define stages the pipeline should run in the given order. Available options are:
    # denoise  - noise reduction
    # detect   - detect and extract signal from data (cut leading and tailing silence)
    # resample - resample signal length
    #            This also includes the boost stage since after resampling all signals have the same length.
    # boost    - convert all signals to a single big numpy matrix
    #            This increases the speed of the following stages but can also drastically increase the memory usage
    #            since all signals are artificially brought to a uniform length by adding zeros.
    # abs      - absolute value - `np.abs(signal)`
    # outlier  - reduce outliers
    # outlier2 - reduce outlier until no outliers are detected anymore
    # smooth   - smooth the signal
    # norm     - normalize
    # double_integration - tapers, filters and integrates signal twice
    # STAGES: list[str] = ["denoise", "detect", "resample", "smooth", "norm"]
    # STAGES: list[str] = ["denoise", "detect", "resample"]
    # STAGES: list[str] = ["denoise", "detect", "resample", "outlier", "norm"]
    STAGES: list[str] = ["denoise", "detect"]
    # STAGES: list[str] = []

    # Define the output format of the processed data
    # npy  - return the data as numpy array
    # pt   - return the data as torch vector
    # list - return the data as list of lists    TODO: not implemented yet
    # hdf5 - return hdf5 files                   TODO: not implemented yet
    # asdf - return asdf files                   TODO: not implemented yet
    # pd   - return pandas files                 TODO: not implemented yet
    # nc   - return xarray files                 TODO: not implemented yet
    # pkl  - return pickle
    OUT_TYPE: str = "pkl"

    # Define the output directory where the file will be written to. Available options are:
    # string - specify the specific folder
    OUT_DIR: str = "./results/"

    # Define the output name of the file (without file extension).
    # OUT_NAME: str = "my-result-data"
    OUT_NAME: str = f"InferenzDaten20250606-{SENSOR}-{SENSOR_DATA}-{AXIS}-{''.join({'detect': 'trim'}.get(stage, stage)[0].lower() for stage in STAGES) or 'raw'}"  # noqa: E501

    # Define if only metadata and no sample/singal data should be loaded.
    LOAD_METADATA_ONLY: bool = False

    # Define the seed for the random number generator.
    # None if no seed should be set, otherwise an integer.
    SEED: int | None = 42

    # Define if signals should be skipped
    # if the timestamps differ to much from the product of sample rate and sample length.
    SKIP_TIMESTAMP_MISSMATCH: bool = False

    # ################################
    # ###### SIGNAL PARAMETERS #######
    # ################################

    # Define a fix (or overwrite) the sample rate (in Hz) for all signals.
    # If None, the sample rate provided from the data is used.
    SAMPLE_RATE: float | None = None
    # NOTE: It is also possible to transform time series from there original SR to a specific target SR.
    #       see: librosa.resample

    # Define the minimal (lower bound) and maximal (upper bound) sample length of the signal.
    # All signals with a lengths outside these boundaries will be skipped.
    SAMPLE_LENGTH_LOWER_BOUND: int | None = 10000
    SAMPLE_LENGTH_UPPER_BOUND: int | None = 2500000  # 1000000

    # Define the minimal (lower bound) and maximal (upper bound) sample rate of the signal.
    # All signals with a sample rate outside these boundaries will be skipped.
    SAMPLE_RATE_LOWER_BOUND: int | None = 10000
    SAMPLE_RATE_UPPER_BOUND: int | None = 15000
    # NOTE: If you chose the upper bound to high, you may need to adjust `time_mask_smooth_ms` in `DENOISE_OPTIONS`

    # #########################################
    # ####### NORMALIZATION PARAMETERS ########
    # #########################################

    # Define the normalization type of the "norm" stage. Available options are:
    # mean_var - Scales time series so that their mean (resp. standard deviation) in each dimension is mu (resp. std).
    # zero_one - Scales time series so that their span in each dimension is between zero and one.
    # mone_one - Scales time series so that their span in each dimension is between minus one and one.
    # mone_one_zero_fix - Like "mone_one" but keeps zeros fix at zero.
    # l1       - Scales time series so that their L1 norm in each dimension is one.
    # l2       - Scales time series so that their L2 norm in each dimension is one.
    # max      - Scales time series so that their Inf norm in each dimension is one.
    NORM_TYPE: TNorm = "mone_one_zero_fix"

    # #########################################
    # ###### NOISE REDUCTION PARAMETERS #######
    # #########################################

    # https://github.com/timsainb/noisereduce/blob/master/noisereduce/noisereduce.py
    # If no "sr" attribute is provided or set to None, the sample rate (sr) from the time series data will be used.
    DENOISE_OPTIONS: dict[str, Any] = dict(
        n_jobs=-1,  # -1 to use all CPU cores, but not possible in combination with use_torch=True
        use_tqdm=False,  # since it is already parallelized, we can not show a additional (parallelized) progress bar
        # use_torch=True,
        # device="cuda:1",
        # time_mask_smooth_ms=250,
    )

    # #########################################
    # ######### SMOOTHING PARAMETERS ##########
    # #########################################

    # Define the smoothing type of the "smooth" stage. Available options are:
    # conv - ConvolutionSmoother
    # exp  - ExponentialSmoother
    SMOOTH_TYPE: TSmoothType = "conv"

    # Define the smoothing parameters of the "smooth" stage.
    # Please refer to the documentation of the corresponding smoother for more information.
    SMOOTHER_KWARGS: dict[str, Any] = {
        # Convolution Smoother
        # "window_length": 10,
        # "window_type": "hanning",
        # Exponential Smoother
        # "window_length": 10,
        # "alpha": 0.5,
    }

    # #########################################
    # ##### SIGNAl EXTRACTION PARAMETERS ######
    # #########################################

    WINDOW_SIZE: int = 100
    VAR_THRESHOLD: float = 0.05
    REMOVE_UNRECOGNIZED_SAMPLES: bool = True

    # #########################################
    # ######### RESAMPLING PARAMETERS #########
    # #########################################

    # Define the resample size of the "resample" stage.
    # The resample size determines the length of the time series.
    RESAMPLE_SIZE: int = 10000

    # #########################################
    # ##### OUTLIER REDUCTION PARAMETERS ######
    # #########################################

    # Define the number of outlier reduction intervals.
    OUTLIER_REDUCTION_INTERVALS: int = 15

    # #########################################
    # ##### DOUBLE INTEGRATION PARAMETERS #####
    # #########################################

    # Define the cutoff filter frequencies in Hz (low_hz, high_hz).
    CUTOFF_FREQ_IN_HZ: tuple[float, float] = (0.5, 15)

    # Define the number of seconds to taper the data array at the beginning and end.
    TAP_S: float | int = 3

    # Define whether to taper the signal with tukey window at beginning only.
    ONE_SIDE_TAPER_FLIPPED: bool = False

    # Define the filter order.
    F_ORDER: int = 2

    # Define whether to convert the double integrated result signal from meters to millimeters.
    CONVERT_M_TO_MM: bool = True

    # #########################################
    # ########## DRAWING PARAMETERS ###########
    # #########################################

    # Define the plot output directory.
    PLOT_OUTPUT_DIR: str = "./results/plots"

    # Define whether to show each plot.
    PLOT_SHOW: bool = False

    # Define whether to save each plot.
    PLOT_SAVE: bool = True

    # Define the plot file formats. Available options are e.g.: png, svg, pdf, pgf, ...
    # All formats supported by matplotlib can be used.
    PLOT_FILE_FORMATS: list[str] = ["png"]

    # #########################################
    # ##### PEAK DETECTION AND CLUSTERING #####
    # #########################################

    # Define peak minimum size to be recognized as a valid peak.
    # Formula: peak_minimum = max(0.05, PEAK_MINIMUM_SIGNAL_FACTOR * np.mean(np.abs(signal[peaks])))
    PEAK_MINIMUM_SIGNAL_FACTOR: float = 0.25

    # Define minimal peak prominence to be recognized as a valid peak.
    MINIMAL_PEAK_PROMINENCE: float = 0.1

    @classmethod
    def peak_distance(cls, current_signal_length: int, original_signal_length: int) -> float | None:
        """
        Required minimal horizontal distance (>= 1) in samples between neighboring peaks.

        Smaller peaks are removed first until the condition is fulfilled for all remaining peaks.

        Also see: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>

        Args:
            current_signal_length (int): The length of the current signal, e.g. after resampling etc.
            original_signal_length (int): The length of the original signal.

        Returns:
            Optional[float]: Minimal horizontal distance or None if not set.
        """
        return 3500 * current_signal_length / original_signal_length

    @classmethod
    def peak_prominence_wlen(cls, current_signal_length: int, original_signal_length: int) -> int | None:
        """
        Window length in samples that optionally limits the evaluated area for each peak to a subset of the signal.

        The peak is always placed in the middle of the window therefore the given length is rounded up
        to the next odd integer.

        Also see: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html>

        Args:
            current_signal_length (int): The length of the current signal, e.g. after resampling etc.
            original_signal_length (int): The length of the original signal.

        Returns:
            Optional[int]: Window length or None if not set.
        """
        d = cls.peak_distance(current_signal_length, original_signal_length)
        return None if d is None else int(d)

    @classmethod
    def cluster_max_sample_distance(cls, current_signal_length: int, original_signal_length: int) -> float:
        """
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.

        This is not a maximum bound on the distances of points within a cluster. This is the most important
        DBSCAN parameter "eps" to choose appropriately for your data set and distance function.

        Also see: <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>

        Args:
            current_signal_length (int): The length of the current signal, e.g. after resampling etc.
            original_signal_length (int): The length of the original signal.

        Returns:
            float: DBSCAN eps value.
        """
        return cls.peak_distance(current_signal_length, original_signal_length) or 0.5

    @classmethod
    def cluster_rect_drawing_delta(cls, current_signal_length: int, original_signal_length: int) -> int:
        """
        Additional size of the rectangle to draw around a cluster.

        While displaying `slki.utils.peak.PeaksClusteringResult` this delta defines the additional size of
        the rectangle to be drawn.

        Args:
            current_signal_length (int): The length of the current signal, e.g. after resampling etc.
            original_signal_length (int): The length of the original signal.

        Returns:
            int: Additional rectangle size.
        """
        eps = cls.cluster_max_sample_distance(current_signal_length, original_signal_length)
        return int(eps // 3) if eps > 3 else 10
