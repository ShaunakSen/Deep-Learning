import cv2 as cv

file_path = 'C:/Users/shaun/Documents/my_projects/Deep-Learning/Computer Vision with Deep Learning/opencv python/Resources/'

### Read image
img = cv.imread(f"{file_path}/Photos/cat.jpg")
# cv.imshow('Cat', img)
# cv.waitKey(0)

### Read Video
"""
can receive a path or integers like 0,1,2
ints are when u want to use web cam or other system cameras
we have to read the video frame by frame
"""
capture = cv.VideoCapture(f'{file_path}/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read() ## returns a frame and a flag that says whether the frame was read or not
    ### display the frame
    if not isTrue:
        break
    cv.imshow('Video', frame)

    ### if letter 'd' is pressed break
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
### release capture point and close all windows
capture.release()
cv.destroyAllWindows()