from types import SimpleNamespace
from enum import Enum
import logging
import cv2

from .detector import SSDDetector, YOLODetector, PublicDetector
from .feature_extractor import FeatureExtractor
from .tracker import MultiTracker
from .utils import Profiler
from .utils.visualization import Visualizer

LOGGER = logging.getLogger(__name__)
class DetectorType(Enum):
    SSD = 0
    YOLO = 1
    PUBLIC = 2

class MOT:
    def __init__(self, size,
                 detector_type='YOLO',
                 detector_frame_skip=5,
                 ssd_detector_cfg=None,
                 yolo_detector_cfg=None,
                 public_detector_cfg=None,
                 feature_extractor_cfg=None,
                 tracker_cfg=None,
                 visualizer_cfg=None,
                 draw=False,
                 number_of_trackers=1):
        """Top level module that integrates detection, feature extraction,
        and tracking together.

        Parameters
        ----------
        size : tuple
            Width and height of each frame.
        detector_type : {'SSD', 'YOLO', 'public'}, optional
            Type of detector to use.
        detector_frame_skip : int, optional
            Number of frames to skip for the detector.
        ssd_detector_cfg : SimpleNamespace, optional
            SSD detector configuration.
        yolo_detector_cfg : SimpleNamespace, optional
            YOLO detector configuration.
        public_detector_cfg : SimpleNamespace, optional
            Public detector configuration.
        feature_extractor_cfg : SimpleNamespace, optional
            Feature extractor configuration.
        tracker_cfg : SimpleNamespace, optional
            Tracker configuration.
        visualizer_cfg : SimpleNamespace, optional
            Visualization configuration.
        draw : bool, optional
            Enable visualization.
        """
        self.size = size
        self.detector_type = DetectorType[detector_type.upper()]
        assert detector_frame_skip >= 1
        self.detector_frame_skip = detector_frame_skip
        self.draw = draw

        self.number_of_trackers = number_of_trackers

        if ssd_detector_cfg is None:
            ssd_detector_cfg = SimpleNamespace()
        if yolo_detector_cfg is None:
            yolo_detector_cfg = SimpleNamespace()
        if public_detector_cfg is None:
            public_detector_cfg = SimpleNamespace()
        if feature_extractor_cfg is None:
            feature_extractor_cfg = SimpleNamespace()
        if tracker_cfg is None:
            tracker_cfg = SimpleNamespace()
        if visualizer_cfg is None:
            visualizer_cfg = SimpleNamespace()

        LOGGER.info('Loading detector model...')
        if self.detector_type == DetectorType.SSD:
            self.detector = SSDDetector(self.size, **vars(ssd_detector_cfg))
        elif self.detector_type == DetectorType.YOLO:
            self.detector = YOLODetector(self.size, **vars(yolo_detector_cfg))
        elif self.detector_type == DetectorType.PUBLIC:
            self.detector = PublicDetector(self.size, self.detector_frame_skip,
                                           **vars(public_detector_cfg))

        LOGGER.info('Loading feature extractor model...')
        self.extractor = FeatureExtractor(**vars(feature_extractor_cfg))

        #self.tracker = MultiTracker(self.size, self.extractor.metric, **vars(tracker_cfg))
        self.trackers = []
        self.frame_counts = []
        self.visualizers = []
        for i in range(self.number_of_trackers):
            self.trackers.append(MultiTracker(self.size, self.extractor.metric, **vars(tracker_cfg)))
            self.frame_counts.append(0)
            self.visualizers.append(Visualizer(**vars(visualizer_cfg))) 

        #self.visualizer = Visualizer(**vars(visualizer_cfg))
        #self.frame_count = 0

    def visible_tracks(self):
        """Retrieve visible tracks from the tracker

        Returns
        -------
        Iterator[Track]
            Confirmed and active tracks from the tracker.
        """
        visible_tracks = []
        for i in range(self.number_of_trackers):
            visible_tracks.append(track for track in self.trackers[i].tracks.values() if track.confirmed and track.active)
        return visible_tracks

    #should I add multiple cap_dt or not???????????
    def reset(self, *streams):
        """Resets multiple object tracker. Must be called before `step`.

        Parameters
        ----------
        cap_dt : float
            Time interval in seconds between each frame.
        """
        self.frame_counts = []
        for i in range(self.number_of_trackers):
            self.trackers[i].reset(streams[i].cap_dt)
            self.frame_counts.append(0)

    def step(self, *frames):
        """Runs multiple object tracker on the next frame.

        Parameters
        ----------
        frame : ndarray
            The next frame.
        """
        ##assuming number of frames and number of trackers are equal!!!!!!!!!!!!!!!!!!!!!!
    
        for i in range(len(frames)):
            frame = frames[i]
            if frame is not None:
                detections = []
                if self.frame_counts[i] == 0:
                    detections = self.detector(frames[i])
                    self.trackers[i].init(frames[i], detections)
                else:
                    if self.frame_counts[i] % self.detector_frame_skip == 0:
                        with Profiler('preproc'):
                            self.detector.detect_async(frames[i])

                        with Profiler('detect'):
                            with Profiler('track'):
                                self.trackers[i].compute_flow(frames[i])
                            detections = self.detector.postprocess()

                        with Profiler('extract'):
                            self.extractor.extract_async(frames[i], detections.tlbr)
                            with Profiler('track', aggregate=True):
                                self.trackers[i].apply_kalman()
                            embeddings = self.extractor.postprocess()

                        with Profiler('assoc'):
                            self.trackers[i].update(self.frame_counts[i], detections, embeddings)
                    else:
                        with Profiler('track'):
                            self.trackers[i].track(frames[i])

                if self.draw:
                    self._draw(frames[i], detections, i)

                self.frame_counts[i] += 1

    @staticmethod
    def print_timing_info():
        LOGGER.debug('=================Timing Stats=================')
        LOGGER.debug(f"{'track time:':<37}{Profiler.get_avg_millis('track'):>6.3f} ms")
        LOGGER.debug(f"{'preprocess time:':<37}{Profiler.get_avg_millis('preproc'):>6.3f} ms")
        LOGGER.debug(f"{'detect/flow time:':<37}{Profiler.get_avg_millis('detect'):>6.3f} ms")
        LOGGER.debug(f"{'feature extract/kalman filter time:':<37}"
                     f"{Profiler.get_avg_millis('extract'):>6.3f} ms")
        LOGGER.debug(f"{'association time:':<37}{Profiler.get_avg_millis('assoc'):>6.3f} ms")

    def _draw(self, frame, detections, tracker_id):
        visible_tracks = list(self.visible_tracks()[tracker_id])
        self.visualizers[tracker_id].render(frame, visible_tracks, detections, self.trackers[tracker_id].klt_bboxes.values(),
                               self.trackers[tracker_id].flow.prev_bg_keypoints, self.trackers[tracker_id].flow.bg_keypoints)
        cv2.putText(frame, f'visible: {len(visible_tracks)}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2, cv2.LINE_AA)
