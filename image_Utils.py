import cv2
import random 

def image_DrawBox(image, dataBox, rand):
    if rand == True:
        color = (random.randrange(0, 255, 1, int), random.randrange(0, 255, 1, int), random.randrange(0, 255, 1, int))
    else:
        color = (50, 255, 200)
    
    cv2.rectangle(image,(dataBox[0], dataBox[1]), (dataBox[2], dataBox[3]), color,2)
    return image

def image_Write(image, text, dataBox, rand):
    if rand == True:
        color = (random.randrange(0, 255, 1, int), random.randrange(0, 255, 1, int), random.randrange(0, 255, 1, int))
    else:
        color = (50, 255, 200)
    cv2.putText(image, text, (dataBox[0]-5, dataBox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2  )
    return image