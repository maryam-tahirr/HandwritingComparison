import pytesseract
import cv2
import PIL.Image
from pytesseract import Output

"""
Page segmentation modes:
  0    Orientation and script detection (OSD) only.
  1    Automatic page segmentation with OSD.
  2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
  3    Fully automatic page segmentation, but no OSD. (Default)
  4    Assume a single column of text of variable sizes.
  5    Assume a single uniform block of vertically aligned text.
  6    Assume a single uniform block of text.
  7    Treat the image as a single text line.
  8    Treat the image as a single word.
  9    Treat the image as a single word in a circle.
 10    Treat the image as a single character.
 11    Sparse text. Find as much text as possible in no particular order.
 12    Sparse text with OSD.
 13    Raw line. Treat the image as a single text line,
       bypassing hacks that are Tesseract-specific.
"""
myconfig = r"--psm 11 --oem 3"

text = pytesseract.image_to_string(PIL.Image.open('Extracttextfromimages\images.png'), config=myconfig)
print(text)

image = cv2.imread('Extracttextfromimages\images.png')
height, width, _ = image.shape

data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=myconfig)

print(data['text'])

amount_boxes = len(data['text'])
for i in range(amount_boxes):
    if float(data['conf'][i])>60:
        (x,y,width,height) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        image = cv2.rectangle(image, (x,y), (x+width, y+height), (0,255,0), 2)
        image = cv2.putText(image, data['text'][i], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2,cv2.LINE_AA)

cv2.imshow('image', image)
cv2.waitKey(0)