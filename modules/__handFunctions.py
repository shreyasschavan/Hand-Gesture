import time
import pyautogui

class HAND_FUNCTION:
    
    __CONFIDENCE = 0.687644321
    
    def __init__(self, choice = '', delay = 0.0, pos=0) -> None:
        self.__CHOICE = choice
        print(self.__CHOICE)
        self.__DELAY = delay
        pyautogui.FAILSAFE = False
        
    def runPlayPause(self) -> str:
        if self.__CHOICE == 'youtube':
            time.sleep(self.__DELAY)
            pyautogui.press('space')
        if self.__CHOICE == 'vlc':
            pyautogui.press("space")
        return "PLAY/PAUSE"
    
    def runVolumeIncrease(self) -> str:
        if self.__CHOICE == 'youtube':
            time.sleep(self.__DELAY)
            pyautogui.press('up')
        if self.__CHOICE == 'vlc':
            pyautogui.press("up")
        return "VOLUME INCREASE"
    
    def runVolumeDecrease(self) -> str:
        if self.__CHOICE == 'youtube':
            time.sleep(self.__DELAY)
            pyautogui.press('down')
        if self.__CHOICE == 'vlc':
            pyautogui.press("down")
        return "VOLUME DECREASE"
    
    def runForward(self) -> str:
        if self.__CHOICE == 'youtube':
            time.sleep(self.__DELAY)
            pyautogui.press('right')
        if self.__CHOICE == 'vlc':
            pyautogui.press("right")
        return "FORWARD"

    def runBackward(self) -> str:
        if self.__CHOICE == 'youtube':
            time.sleep(self.__DELAY)
            pyautogui.press('left')
        if self.__CHOICE == 'vlc':
            pyautogui.press("left")
        return "BACKWARD"
    
    def runFullScreen(self) -> str:
        if self.__CHOICE == 'youtube':
            time.sleep(self.__DELAY)
            pyautogui.press('f')
        if self.__CHOICE == 'vlc':
            pyautogui.press("f")
        return "FULLSCREEN"
        