from secrets import choice
from urllib.parse import _NetlocResultMixinStr

# from HAND_GESTURE_MEDIAPLAYER_CONTROLER import YT_VLC_CONTROL_HAND_TRACKING
import cv2
import imutils
import mediapipe
import sys

from modules.__hand_tracking import HandTracking
from modules.__handFunctions import HAND_FUNCTION


class MediaPlayer_GestureRecognition:
    """
    WebCam sources
    """
    # __mobile_web_cam_url = 'https://0.0.0.0:8080/video'
    __HCAM, __WCAM = 600, 600
    __START_CAM = True
    __ESCAPE = 'q'
    __FRAME = None

    """
    Min and Max Degree
    """
    __DEGREE_MIN = -130
    __DEGREE_MAX = -60

    """
    Max and Min area of hand
    """
    __MAX_AREA = 900
    __MIN_AREA = 100

    """
    Play Pause FingersUP
    """
    __PLAY_PAUSE_FINGERUP = [0, 1, 1, 1, 0]
    __PLAY_PAUSE_FINGERS = [4, 13]

    """
    Volume FingersUP
    """
    __VOLUME_FINGERUP = [1, 1, 0, 0, 1]
    __VOLUME_FINGERDOWN = [0, 1, 0, 0, 1]

    """
    Forward FingerUP
    """
    __FORWARD_FINGERUP = [0, 1, 1, 0, 0]

    """ 
    Backward FingerUP
    """
    __BACKWARD_FINGERUP = [1, 1, 1, 0, 0]

    """ 
    Fullscreen FingerUP
    """
    __FULLSCREEN_FINGERUP = [0, 0, 0, 0, 0]

    __CURRENT_ACTION = None

    __FRAME_NAME = "Gesture Control"

    __CLICK_THRESHOLD = 10

    __RIGHT_HAND_INDEX = 1
    __LEFT_HAND_INDEX = 0

    def __init__(self, CHOICE='vlc', WEB_CAM_SOURCE=0, swipe=True):
        print("[~] INITIALISING...")
        """
        Init
        """
        self.__CHOICE = CHOICE
        self.__WEB_CAM_SOURCE = WEB_CAM_SOURCE
        self.__SWIPE = swipe

        """
        Hand tracking module inputes
        """
        self.__MIN_DETECTION_CONFIDENCE = 0.75
        self.__MAX_HAND = 1
        self.__MIN_TRACKING_CONFIDENCE = 0.5

        """
        Initialise Modules
        """
        print("[~] INITIALISING WEBCAM....")

        ###WebCam input
        self.__CAP = cv2.VideoCapture(self.__WEB_CAM_SOURCE)
        self.__CAP.set(3, self.__WCAM)
        self.__CAP.set(4, self.__HCAM)

        """
        Initialize HAND TRACKING MODULE
        """
        self.__HAND_TRACK = HandTracking(
            min_detection_confidence=self.__MIN_DETECTION_CONFIDENCE,
            maxHands=self.__MAX_HAND,
            min_tracking_confidence=self.__MIN_TRACKING_CONFIDENCE
        )

        self.__HAND_FUNCTION = HAND_FUNCTION(choice=self.__CHOICE)

    def __del__(self):
        self.__CAP.release()
        cv2.destroyAllWindows()

    def start(self):

        print("\t[~] STARTING WEBCAM....\n")

        while self.__START_CAM:
            __S, self.__FRAME = self.__CAP.read()

            try:
                # resize the frame with width = 600
                self.__FRAME = imutils.resize(self.__FRAME,
                                              width=600)

            except AttributeError as e:
                print("[!] Connect webcam....");sys.e
            if not __S: print('[~] Check WebCam.')

            # Flip on horizontal
            self.__FRAME = cv2.flip(self.__FRAME, 1)

            self.__FRAME, __RESULT = self.__HAND_TRACK.getKeyPointsWithFrame(image=self.__FRAME)

            __PTLIST, __BBOX, __AREA = self.__HAND_TRACK.findPosition(results=__RESULT,
                                                                      image_shape=self.__FRAME.shape)

            # If and detected and right hand
            if len(__PTLIST) != 0 and __RESULT.multi_handedness[0].classification[0].index == self.__RIGHT_HAND_INDEX:

                """
                Draw Hands
                """
                # self.__frame = self.__hand_track.draw_hands(image = self.__frame)
                self.__FRAME = self.__HAND_TRACK.draw(
                    img=self.__FRAME,
                    bbox=__BBOX,
                    draw_hand=True,
                    draw_fancy=False,
                    style_type='borderless'
                )

                """
                List of Fingers
                1 - Up
                0 - Down
                """
                __IS_FINGER_DOWN = self.__HAND_TRACK.fingersUp()
                # print(__IS_FINGER_DOWN)

                """
                If Degree is in range of -130 to -60.
                """
                __HAND_DEGREE = int(self.__HAND_TRACK.findDegree())

                # """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                print(
                    f"[~](LOG): FINGER UP/DOWN: {__IS_FINGER_DOWN}, AREA: {__BBOX}, DEGREE: {__HAND_DEGREE} LAST ACTION: {self.__CURRENT_ACTION}")
                # """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

                # print(self.__DEGREE_MIN, self.__DEGREE_MAX, __degree, self.__DEGREE_MIN < __degree < self.__DEGREE_MAX)

                if (self.__DEGREE_MIN < __HAND_DEGREE < self.__DEGREE_MAX):

                    """
                    Play Pause
                    """
                    if __IS_FINGER_DOWN == self.__PLAY_PAUSE_FINGERUP:
                        __isPlayPauseClicked = self.__HAND_TRACK.makeClick(fingerNO=self.__PLAY_PAUSE_FINGERS[0],
                                                                           fingerNO2=self.__PLAY_PAUSE_FINGERS[1],
                                                                           max_len=self.__CLICK_THRESHOLD * 2)
                        if __isPlayPauseClicked:
                            self.__CURRENT_ACTION = self.__HAND_FUNCTION.runPlayPause()
                            __isPlayPauseClicked = False

                    """
                    Volume UP
                    """
                    if __IS_FINGER_DOWN == self.__VOLUME_FINGERUP:
                        __isVolumeUP = self.__HAND_TRACK.makeClick(is_single_finger=True,
                                                                   max_len=self.__CLICK_THRESHOLD)
                        if __isVolumeUP:
                            self.__CURRENT_ACTION = self.__HAND_FUNCTION.runVolumeIncrease()
                            __isVolumeUP = False

                    """
                    Volume DOWN
                    """
                    if __IS_FINGER_DOWN == self.__VOLUME_FINGERDOWN:
                        __isVolumeDOWN = self.__HAND_TRACK.makeClick(is_single_finger=True,
                                                                     max_len=self.__CLICK_THRESHOLD)
                        if __isVolumeDOWN:
                            self.__CURRENT_ACTION = self.__HAND_FUNCTION.runVolumeDecrease()
                            __isVolumeDOWN = False

                    """
                    Forward 
                    """
                    if __IS_FINGER_DOWN == self.__FORWARD_FINGERUP:
                        __isForward = self.__HAND_TRACK.makeClick(is_single_finger=True,
                                                                  max_len=self.__CLICK_THRESHOLD)
                        if __isForward:
                            self.__CURRENT_ACTION = self.__HAND_FUNCTION.runForward()
                            __isForward = False

                    """
                    Backward or Rewind
                    """
                    if __IS_FINGER_DOWN == self.__BACKWARD_FINGERUP:
                        __isBackward = self.__HAND_TRACK.makeClick(is_single_finger=True,
                                                                   max_len=self.__CLICK_THRESHOLD)
                        if __isBackward:
                            self.__CURRENT_ACTION = self.__HAND_FUNCTION.runBackward()
                            __isBackward = False

                    """
                    Fullscreen
                    """
                    if __IS_FINGER_DOWN == self.__FULLSCREEN_FINGERUP:
                        __isFullScreen = self.__HAND_TRACK.makeClick(is_single_finger=True,
                                                                     max_len=self.__CLICK_THRESHOLD * 2)
                        if __isFullScreen:
                            self.__CURRENT_ACTION = self.__HAND_FUNCTION.runFullScreen()
                            __isFullScreen = False

            self.__FRAME = self.__HAND_TRACK.writeText(self.__FRAME,
                                                       self.__CURRENT_ACTION, None)

            cv2.imshow(self.__FRAME_NAME,
                       self.__FRAME)

            self.__check_exit()

    def __check_exit(self):
        if cv2.waitKey(1) == ord(self.__ESCAPE):
            self.__START_CAM = False


if __name__ == "__main__":
    app = MediaPlayer_GestureRecognition(CHOICE='vlc')
    app.start()
