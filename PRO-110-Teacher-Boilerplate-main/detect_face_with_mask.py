# import the opencv library                
import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model("keras_model.h5") 
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    #resizing the image
    img = cv2.resize(frame,(224,224))
    #converting the image into numpy array 
    test_image = np.array(img, dtype = np.float32)
    test_image = np.expand_dins(test_image, axis=0) 
    #normalizing the image
    normalize_image=test_image/255.0
    #prediction result
    prediction = model.predict(frame)
    print("prediction ",prediction)
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()

