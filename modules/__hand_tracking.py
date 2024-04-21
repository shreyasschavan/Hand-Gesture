import mediapipe as mp
import cv2
import numpy as np
import math
from modules.__colors import COLORS
import random

class HandTracking:
    __TIP_NO = [4, 8, 12, 16, 20]
    
    """
    Colors
    """
    __BOUNDRY_COLORS = [COLORS.RED,
                      COLORS.PINK,
                      COLORS.GREEN,
                      COLORS.YELLOW,
                      COLORS.WHITE]
    __BOUNDR_FANCY_COLOR = (0, 0, 0)
    
    
    """
    OpenCv text font style
    """
    __TEXT_FONT = cv2.FONT_HERSHEY_PLAIN
    __TEXT_POS = (30, 40)
    __FONT_SIZE = 3
    
    
    """
    Radius
    """
    __RADIUS = 5
    __LINE_THICKNESS = 1

    """
    Thickness
    """
    __BOUNDRY_THICKNESS = 1
    __LINE_THICKNESS = 1
    __FANCY_THICKNESS = 10
    
    __LINES = [[0, 17],
             [17, 20],
             [0, 13],
             [13, 16],
             [0, 9],
             [9, 12],
             [0, 5],
             [5, 8],
             [0, 2],
             [2, 4]]
    
    """
    CALCULATING CLICK
    """
    __XPOS, __YPOS = [], []
    __NEW_FINGERNO, __NEW_FINGERNO2 = None, None
    
    """
    Thresholds
    """
    __BOUNDRY_THRESH = 20
    __FANCY_THRESH = 30
    __FANCY_THRESH_BORDER = 15
    
    
    """
    Style Data
    """
    __STYLE_TYPE_DATA = {
            "borderless" : 0, 
            "with_border" : 1, 
            "with_border_2.0" : 2
        }

    def __init__(self, min_detection_confidence = 0.5, 
                 min_tracking_confidence = 0.5,
                 image_mode = False,
                 maxHands = 1) -> None:

        """
        min_detection_confidence : MINIMUM DETECTION CONFIDENCE (Default = 0.5)
        min_tracking_confidence  : MINIMUM TRACKING CONFIDENCE (Default = 0.5)
        image_mode               : static_image_mode (Default = False)
        """
        
        self.__image_mode = image_mode
        self.maxHands = maxHands

        self.__mp_drawing = mp.solutions.drawing_utils
        self.__mp_hands = mp.solutions.hands

        self.__min_detection_confidence = min_detection_confidence
        self.__min_tracking_confidence = min_tracking_confidence
        
        self.__hands = self.__mp_hands.Hands(min_detection_confidence=self.__min_detection_confidence, 
                            min_tracking_confidence=self.__min_tracking_confidence,
                            max_num_hands=self.maxHands,
                            static_image_mode=self.__image_mode)

    def getKeyPointsWithFrame(self, image) -> np.ndarray:

        """
        Convert Frame to BGR to RGB
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        """
        Process the Current Frame
        """
        self.results = self.__hands.process(image)
        image.flags.writeable = True

        """
        Convert back to RGB
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)     
        return image, self.results

    def findPosition(self, results, image_shape, handNo = 0) -> list:
             
        self.__lmList = []
        __XList, __YList = [], []

        __bbox = ()
        __area = 0

        if results.multi_hand_landmarks:
            __myHand = results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(__myHand.landmark):
                h, w, c = image_shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                __XList.append(cx)
                __YList.append(cy)
                self.__lmList.append([id, cx, cy])
            xmin, xmax = min(__XList), max(__XList)
            ymin, ymax = min(__YList), max(__YList)
            __bbox = (xmin, 
                    ymin,
                    xmax,
                    ymax)
            
            __area = (__bbox[2] - __bbox[0]) * (__bbox[3] - __bbox[1]) // 100
            
        return self.__lmList, __bbox, __area 
     
    def __draw_fancy(self, style_type, img, BOUNDR_FANCY_COLOR, bbox) -> None:
    
        x, y, w, h = bbox
        x -= self.__BOUNDRY_THRESH
        y -= self.__BOUNDRY_THRESH
        w += self.__BOUNDRY_THRESH
        h += self.__BOUNDRY_THRESH

        if self.__STYLE_TYPE_DATA[style_type] == 0:
                        
            """
            Top left
            """
            cv2.line(img,
                    (x, y), 
                    (x + self.__FANCY_THRESH, y),
                    BOUNDR_FANCY_COLOR, 
                    self.__FANCY_THICKNESS)
            cv2.line(img, 
                    (x, y), 
                    (x, y + self.__FANCY_THRESH),
                    BOUNDR_FANCY_COLOR, 
                    self.__FANCY_THICKNESS)
            
            """
            Top right
            """
            cv2.line(img,
                    (w - self.__BOUNDRY_THRESH - self.__FANCY_THICKNESS, y),
                    (w, y),
                    BOUNDR_FANCY_COLOR,
                    self.__FANCY_THICKNESS)
            cv2.line(img, (w + self.__BOUNDRY_THRESH - self.__FANCY_THICKNESS*2, y),
                    (w + self.__BOUNDRY_THRESH - self.__FANCY_THICKNESS*2, y + self.__FANCY_THRESH),
                    BOUNDR_FANCY_COLOR,
                    self.__FANCY_THICKNESS)
            
            """
            Bottom left
            """
            cv2.line(img,
                    (x, h), 
                    (x + self.__BOUNDRY_THRESH, h),
                    BOUNDR_FANCY_COLOR, 
                    self.__FANCY_THICKNESS)
            cv2.line(img, 
                    (x, h),
                    (x, h - self.__BOUNDRY_THRESH),
                    BOUNDR_FANCY_COLOR, 
                    self.__FANCY_THICKNESS)

            
            """
            Bottom right
            """
            cv2.line(img,
                    (w - self.__BOUNDRY_THRESH - self.__FANCY_THICKNESS, h),
                    (w, h),
                    BOUNDR_FANCY_COLOR,
                    self.__FANCY_THICKNESS)
            cv2.line(img, (w + self.__BOUNDRY_THRESH - self.__FANCY_THICKNESS*2, h),
                    (w + self.__BOUNDRY_THRESH - self.__FANCY_THICKNESS*2, h - self.__FANCY_THRESH),
                    BOUNDR_FANCY_COLOR,
                    self.__FANCY_THICKNESS)
            
        if self.__STYLE_TYPE_DATA[style_type] == 1:         
                        
            """
            Top left
            """
            cv2.line(img,
                    (x - self.__FANCY_THRESH_BORDER, y - self.__FANCY_THRESH_BORDER),
                    (x + self.__FANCY_THRESH, y - self.__FANCY_THRESH_BORDER),
                    BOUNDR_FANCY_COLOR, 
                    self.__FANCY_THICKNESS)
            cv2.line(img, 
                    (x - self.__FANCY_THRESH_BORDER, y - self.__FANCY_THRESH_BORDER),
                    (x - self.__FANCY_THRESH_BORDER, y + self.__FANCY_THRESH),
                    BOUNDR_FANCY_COLOR, 
                    self.__FANCY_THICKNESS)
            
            """
            Top right
            """
            cv2.line(img,
                    (w - self.__BOUNDRY_THRESH - self.__FANCY_THRESH + self.__FANCY_THRESH_BORDER, y - self.__FANCY_THRESH_BORDER),
                    (w + self.__FANCY_THRESH_BORDER, y - self.__FANCY_THRESH_BORDER),
                    BOUNDR_FANCY_COLOR,
                    self.__FANCY_THICKNESS)
            cv2.line(img,
                    (w + self.__BOUNDRY_THRESH - self.__FANCY_THICKNESS*2 + self.__FANCY_THRESH_BORDER, y - self.__FANCY_THRESH_BORDER),
                    (w + self.__BOUNDRY_THRESH - self.__FANCY_THICKNESS*2 + self.__FANCY_THRESH_BORDER, y + self.__FANCY_THRESH),
                    BOUNDR_FANCY_COLOR,
                    self.__FANCY_THICKNESS)
            
            """
            Bottom left
            """
            cv2.line(img,
                    (x - self.__FANCY_THRESH_BORDER, h + self.__FANCY_THRESH_BORDER),
                    (x + self.__FANCY_THRESH, h + self.__FANCY_THRESH_BORDER),
                    BOUNDR_FANCY_COLOR, 
                    self.__FANCY_THICKNESS)
            cv2.line(img, 
                    (x - self.__FANCY_THRESH_BORDER, h - self.__FANCY_THRESH_BORDER*2),
                    (x - self.__FANCY_THRESH_BORDER, h + self.__FANCY_THRESH_BORDER),
                    BOUNDR_FANCY_COLOR, 
                    self.__FANCY_THICKNESS)
            
            """
            Bottom right
            """
            cv2.line(img,
                    (w - self.__BOUNDRY_THRESH - self.__FANCY_THRESH + self.__FANCY_THRESH_BORDER, h + self.__FANCY_THRESH_BORDER),
                    (w + self.__FANCY_THRESH_BORDER, h + self.__FANCY_THRESH_BORDER),
                    BOUNDR_FANCY_COLOR,
                    self.__FANCY_THICKNESS)
            cv2.line(img, 
                    (w + self.__BOUNDRY_THRESH - self.__FANCY_THICKNESS*2 + self.__FANCY_THRESH_BORDER, h - self.__FANCY_THRESH_BORDER*2),
                    (w + self.__BOUNDRY_THRESH - self.__FANCY_THICKNESS*2 + self.__FANCY_THRESH_BORDER, h + self.__FANCY_THRESH_BORDER),
                    BOUNDR_FANCY_COLOR,
                    self.__FANCY_THICKNESS)

        if self.__STYLE_TYPE_DATA[style_type] == 2:
            """
            Top left
            """
            cv2.line(img,
                    (x, y), 
                    (x + self.__FANCY_THRESH, y),
                    BOUNDR_FANCY_COLOR, 
                    self.__FANCY_THICKNESS)
            cv2.line(img, 
                    (x, y), 
                    (x, y + self.__FANCY_THRESH),
                    BOUNDR_FANCY_COLOR, 
                    self.__FANCY_THICKNESS)
            """
            Top right
            """
            cv2.line(img,
                    (w - self.__BOUNDRY_THRESH - self.__FANCY_THICKNESS, y),
                    (w, y),
                    BOUNDR_FANCY_COLOR,
                    self.__FANCY_THICKNESS)
            cv2.line(img, (w + self.__BOUNDRY_THRESH - self.__FANCY_THICKNESS*2, y),
                    (w + self.__BOUNDRY_THRESH - self.__FANCY_THICKNESS*2, y + self.__FANCY_THRESH),
                    BOUNDR_FANCY_COLOR,
                    self.__FANCY_THICKNESS)
            
            """
            Bottom left
            """
            cv2.line(img,
                    (x, h), 
                    (x + self.__BOUNDRY_THRESH + self.__FANCY_THICKNESS, h),
                    BOUNDR_FANCY_COLOR, 
                    self.__FANCY_THICKNESS)
            cv2.line(img, 
                    (x, h),
                    (x, h - self.__BOUNDRY_THRESH - self.__FANCY_THICKNESS),
                    BOUNDR_FANCY_COLOR, 
                    self.__FANCY_THICKNESS)

            
            """
            Bottom right
            """
            cv2.line(img,
                    (w - self.__BOUNDRY_THRESH - self.__FANCY_THICKNESS, h),
                    (w, h),
                    BOUNDR_FANCY_COLOR,
                    self.__FANCY_THICKNESS)
            cv2.line(img, (w + self.__BOUNDRY_THRESH - self.__FANCY_THICKNESS*2, h),
                    (w + self.__BOUNDRY_THRESH - self.__FANCY_THICKNESS*2, h - self.__FANCY_THRESH),
                    BOUNDR_FANCY_COLOR,
                    self.__FANCY_THICKNESS)

    def draw(self, 
             img = None,
             bbox = None,
             draw_boundry = False,
             draw_hand = False,
             draw_fancy = False,
             is_swipe = False,
             style_type = 'borderless',
             is_swipe_type = None) -> list:
        
        """
        if draw_fancy == True
        style_type = [borderless, with_border, with_border_2.0]
        """

        def __get_colors(l1=None,
               l2=None,
               color_mode='', 
               tip_no=None):
            COLOR = (255, 255, 255)
            if color_mode == 'line':
                if l1 == 0 \
                and l2 == 17:                
                    COLOR = COLORS.RED
                if l1 == 17 \
                and l2 == 20:
                    COLOR = COLORS.RED
                if l1 == 0 \
                and l2 == 13:
                    COLOR = COLORS.PINK
                if l1 == 13 \
                and l2 == 16:
                    COLOR =COLORS.PINK
                if l1 == 0 \
                and l2 == 9:
                    COLOR = COLORS.GREEN
                if l1 == 9 \
                and l2 == 12:
                    COLOR = COLORS.GREEN
                if l1 == 0 \
                and l2 == 5:
                    COLOR = COLORS.YELLOW
                if l1 == 5 \
                and l2 == 8:
                    COLOR = COLORS.YELLOW
                if l1 == 0 \
                and l2 == 2:
                    COLOR = COLORS.WHITE
                if l1 == 2 \
                and l2 == 4:
                    COLOR = COLORS.WHITE
                return COLOR
            
            if color_mode == 'tip':
                if tip_no == 20:
                    COLOR = COLORS.RED
                if tip_no == 16:
                    COLOR = COLORS.PINK
                if tip_no == 12:
                    COLOR = COLORS.GREEN
                if tip_no == 8:
                    COLOR = COLORS.YELLOW
                if tip_no == 4:
                    COLOR = COLORS.WHITE
                return COLOR
    
        def __get_random_boundry_color():
            return random.choice(self.__BOUNDRY_COLORS)

        if style_type == "borderless":
            draw_boundry = False
        
        else:#if style_type == "with_border" or style_type == "with_border_2.0":
            draw_boundry = True
                 
        __BOUNDR_FANCY_COLOR = __get_random_boundry_color()

        if draw_boundry:
            x, y, w, h = bbox
           
            cv2.rectangle(img, 
                        (x - self.__BOUNDRY_THRESH, y - self.__BOUNDRY_THRESH),
                        (w + self.__BOUNDRY_THRESH, h + self.__BOUNDRY_THRESH),
                        __BOUNDR_FANCY_COLOR,
                        self.__BOUNDRY_THICKNESS)
        
        if draw_fancy:
            self.__draw_fancy(style_type = style_type, img=img, BOUNDR_FANCY_COLOR=__BOUNDR_FANCY_COLOR, bbox=bbox)
            
        if draw_hand:
            """
            Draw Lines
            """
            for l1, l2 in self.__LINES:
                x1, y1 = self.__lmList[l1][1], \
                    self.__lmList[l1][2]
                x2, y2 = self.__lmList[l2][1],\
                    self.__lmList[l2][2]
                __LINE_COLOR = __get_colors(l1=l1,
                                        l2=l2, 
                                        color_mode = 'line')
                cv2.line(img, 
                            (x1, y1),
                            (x2, y2),
                            __LINE_COLOR,
                            self.__LINE_THICKNESS)
        
                """
                Draw Circle on Tip
                """
                for tip in self.__TIP_NO:
                    __TIP_COLOR = __get_colors(l1=l1,
                                        l2=l2,
                                        color_mode = 'tip', 
                                        tip_no=tip)
                    x, y = self.__lmList[tip][1], \
                        self.__lmList[tip][2]
                    cv2.circle(img,
                                (x, y),
                                self.__RADIUS,
                                __TIP_COLOR,
                                cv2.FILLED)   
        return img
    
    def draw_hands(self, image):
        
        def __get_colors(l1=None,
               l2=None,
               color_mode='', 
               tip_no=None):
            
            COLOR = (255, 255, 255)
            
            if color_mode == 'line':
                if l1 == 0 \
                and l2 == 17:                
                    COLOR = COLORS.RED
                if l1 == 17 \
                and l2 == 20:
                    COLOR = COLORS.RED
                if l1 == 0 \
                and l2 == 13:
                    COLOR = COLORS.PINK
                if l1 == 13 \
                and l2 == 16:
                    COLOR =COLORS.PINK
                if l1 == 0 \
                and l2 == 9:
                    COLOR = COLORS.GREEN
                if l1 == 9 \
                and l2 == 12:
                    COLOR = COLORS.GREEN
                if l1 == 0 \
                and l2 == 5:
                    COLOR = COLORS.YELLOW
                if l1 == 5 \
                and l2 == 8:
                    COLOR = COLORS.YELLOW
                if l1 == 0 \
                and l2 == 2:
                    COLOR = COLORS.WHITE
                if l1 == 2 \
                and l2 == 4:
                    COLOR = COLORS.WHITE
                return COLOR
            if color_mode == 'tip':
                if tip_no == 20:
                    COLOR = COLORS.RED
                if tip_no == 16:
                    COLOR = COLORS.PINK
                if tip_no == 12:
                    COLOR = COLORS.GREEN
                if tip_no == 8:
                    COLOR = COLORS.YELLOW
                if tip_no == 4:
                    COLOR = COLORS.WHITE
                return COLOR

        """
        Draw Lines
        """
        for l1, l2 in self.__LINES:
            x1, y1 = self.__lmList[l1][1], \
                self.__lmList[l1][2]
            x2, y2 = self.__lmList[l2][1],\
                self.__lmList[l2][2]
            __LINE_COLOR = __get_colors(l1=l1,
                                    l2=l2, 
                                    color_mode = 'line')
            cv2.line(image, 
                        (x1, y1),
                        (x2, y2),
                        __LINE_COLOR,
                        self.__LINE_THICKNESS)
    
            """
            Draw Circle on Tip
            """
            for tip in self.__TIP_NO:
                __TIP_COLOR = __get_colors(l1=l1,
                                    l2=l2,
                                    color_mode = 'tip', 
                                    tip_no=tip)
                x, y = self.__lmList[tip][1], \
                    self.__lmList[tip][2]
                cv2.circle(image,
                            (x, y),
                            self.__RADIUS,
                            __TIP_COLOR,
                            cv2.FILLED)  
            
            """ 
            for i in range(len(self.__lmList)):
                x, y = self.__lmList[i][1], \
                    self.__lmList[i][2]
                cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (120, 240, 60))
                cv2.circle(image,
                            (x, y),
                            self.__RADIUS,
                            (120, 120, 120),
                            cv2.FILLED)  
            """
                
                
        return image
    
    def fingersUp(self) -> list:
        __fingers = []

        """
        Thumb Finger
        """
        if self.__lmList[self.__TIP_NO[0]][1] < self.__lmList[self.__TIP_NO[0] - 1][1]:
            __fingers.append(1)
        else:
            __fingers.append(0)

        """
        For Other Fingers
        """
        for id in range(1, 5):
            if self.__lmList[self.__TIP_NO[id]][2] < self.__lmList[self.__TIP_NO[id] - 2][2]:
                __fingers.append(1)
            else:
                __fingers.append(0)
        return __fingers
    
    def findDegree(self) -> float:
        x1, y1 = self.__lmList[0][1], self.__lmList[0][2]
        x2, y2 = self.__lmList[12][1], self.__lmList[12][2]
        __atan2 = math.atan2(y2 - y1, x2 - x1)
        return math.degrees(__atan2)  
    
    def getFindegCoordinate(self, lmList, fingerNo) -> tuple:return (lmList[fingerNo][1:])
    
    def __addPos(self, max_len) -> bool:
        __IS_CLICKED = False
        self.__XPOS.append(1)
        self.__YPOS.append(1)
        print(len(self.__XPOS) and len(self.__YPOS))
        if (len(self.__XPOS) and len(self.__YPOS)) > max_len:
            #print("CLICKED\a",self.__NEW_FINGERNO)
            __IS_CLICKED = True
            self.__XPOS.clear()
            self.__YPOS.clear()
            self.__NEW_FINGERNO = None
            self.__NEW_FINGERNO2 = None
            print("__IS_CLICKED: ",__IS_CLICKED)
            return __IS_CLICKED
        print("__IS_CLICKED2: ",__IS_CLICKED)
        return __IS_CLICKED

    def makeClick(self, fingerNO=None, fingerNO2=None, is_single_finger=False, max_len=30) -> bool:
        __IS_CLICKED = False
        if not is_single_finger:
            if fingerNO == self.__NEW_FINGERNO and fingerNO2 == self.__NEW_FINGERNO2:
                self.__NEW_FINGERNO = fingerNO
                self.__NEW_FINGERNO2 = fingerNO2
                x1, y1 = self.getFindegCoordinate(lmList=self.__lmList, fingerNo=fingerNO)
                x2, y2 = self.getFindegCoordinate(lmList=self.__lmList, fingerNo=fingerNO2)
                print(x1+y1, x2+y2, x1+y1 >= x2+y2)
                if x1+y1 >= x2+y2:
                    return self.__addPos(max_len)
            else:
                __IS_CLICKED = False
                self.__XPOS.clear()
                self.__YPOS.clear()

                if (self.__NEW_FINGERNO or self.__NEW_FINGERNO2) == None:
                    self.__NEW_FINGERNO = fingerNO
                    self.__NEW_FINGERNO2 = fingerNO2
        if is_single_finger:
            return self.__addPos(max_len)
        return __IS_CLICKED
    
    def writeText(self, frame=None, text=None, pos=None):
       # for __colors in self.__BOUNDRY_COLORS:
        cv2.putText(frame, str(text), pos, self.__TEXT_FONT, self.__FONT_SIZE, random.choice(self.__BOUNDRY_COLORS)) if pos != None else self.writeText(frame, text, self.__TEXT_POS)
        return frame
