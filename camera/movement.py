import cv2
import numpy as np
import time
import logging
import signal


class MotionDetector:
    MOVEMENT_TRESHOLD = 20000000
    RECORD_MIN_FRAMES = 10
    WARM_UP_FRAMES = 100

    def __init__(self, debug=False, log_level=logging.INFO):
        self.cap = cv2.VideoCapture(0)

        logging.basicConfig(
            format='[%(asctime)s][%(levelname)s] %(message)s', level=log_level)
        logging.info("Starting camera")

        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        self.previous_frame = None
        self.frame_count = 0
        self.record = False
        self.record_at_frame = None
        self.debug = debug
        self.should_stop = False

        signal.signal(signal.SIGINT, self.graceful_shutdown)

    def start(self):
        while True:
            self.frame_count += 1

            ret, frame = self.cap.read()

            if ret is False:
                continue

            current_frame = self.prepare_frame(frame)

            if (self.previous_frame is None):
                self.previous_frame = current_frame
                continue

            if self.frame_count == self.WARM_UP_FRAMES:
                logging.info(
                    f'Initialisation done, {self.WARM_UP_FRAMES} frames skipped')

            diff_frame = self.create_diff(current_frame)

            self.start_recording(diff_frame)
            self.record_frame(frame)
            self.stop_recording(diff_frame)

            self.previous_frame = current_frame

            if self.debug is True and self.frame_count == self.WARM_UP_FRAMES + 10:
                break

            if self.should_stop is True:
                logging.info("Camera stopped")
                break

    def graceful_shutdown(self, signum, frame):
        logging.info(
            f'Caught {signal.Signals(signum).name} at {frame}, stopping camera')
        self.should_stop = True

    def record_frame(self, frame):
        if self.record is True:
            cv2.imwrite(f'{int(time.time())}_{self.frame_count}.png', frame)

    def start_recording(self, diff_frame):
        # Give the camera some time to start, only
        # recognise movement after couple seconds.
        if self.frame_count > self.WARM_UP_FRAMES and diff_frame.sum() > self.MOVEMENT_TRESHOLD:
            if self.record is False:
                logging.info(f'Started recording at frame {self.frame_count}')

            logging.debug(
                f'Recording state record={self.record} diff_frame={diff_frame.sum()}')
            self.record = True
            self.record_at_frame = self.frame_count

            if self.debug is True and self.frame_count > self.WARM_UP_FRAMES:
                cv2.imwrite(f'{self.frame_count}_20_diff.png', diff_frame)

    def stop_recording(self, diff_frame):
        if self.record is False:
            return

        if diff_frame.sum() < self.MOVEMENT_TRESHOLD and self.frame_count > self.record_at_frame + self.RECORD_MIN_FRAMES:
            logging.info(f'Stopped recording at frame {self.record_at_frame}')
            self.record = False
            self.record_at_frame = None

    def prepare_frame(self, frame):
        img_rgb = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
        prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        blured_frame = cv2.GaussianBlur(
            src=prepared_frame, ksize=(5, 5), sigmaX=0)

        if self.debug is True and self.frame_count > self.WARM_UP_FRAMES:
            cv2.imwrite(f'{self.frame_count}_10_blur.png', blured_frame)

        return blured_frame

    def create_diff(self, current_frame):
        diff_frame = cv2.absdiff(src1=self.previous_frame, src2=current_frame)
        kernel = np.ones((5, 5))
        diff_frame = cv2.dilate(diff_frame, kernel, 1)

        if self.debug is True and self.frame_count > self.WARM_UP_FRAMES:
            cv2.imwrite(f'{self.frame_count}_15_prev.png', self.previous_frame)

        return cv2.threshold(
            src=diff_frame, thresh=20, maxval=255, type=cv2.THRESH_BINARY)[1]


if (__name__ == "__main__"):
    MotionDetector(log_level=logging.DEBUG).start()
