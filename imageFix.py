import cv2
import matplotlib.pyplot as plt
import numpy as np

def apply_fix(image, id):
        #The third image - no correction possible
        if id==3:
                return image
        
        #Calculate the histogram and the comulated histogram
        histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        cum_hist = np.cumsum(histogram)
        '''
        #show the histogram to get the values [u1,u2] for contrast
        plt.plot(histogram, color='b')
        plt.title('Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()
        '''

        #id=0 because there is no picture need histogram equalization, so we will not enter this if
        #I provided two codes to calculate it, the second one by using a cv2 function equalizeHist(image). Both of them return the same result.
        if id == 0:
            # Histogram Equalization

            '''
            # Check if cum_hist.max() is zero to avoid division by zero
            if cum_hist.max() == 0:
                print(f"Error: Cumulative histogram is zero for image {id}")
                return

            #calculate histogram equalization
            cdf_normalized = cum_hist / cum_hist.max()
            image_equalized = np.interp(image.flatten(), range(256), cdf_normalized * 255).reshape(image.shape).astype(np.uint8)
            return image_equalized
            '''
            
            #using cv2 function equalizeHist(image)
            return cv2.equalizeHist(image)
        # ------------------------------------------------------------------------------------------------------------------
        elif id == 2:          
            # Gamma Correction

            # Typical value for gamma is 1/2.2
            gamma = 0.6
            # Convert the image to a NumPy array
            image_array = np.array(image)
            
            # Apply gamma correction
            # Dividing by 255.0 and then multiplying by 255.0 is to ensure that the pixel values remain in the valid range of [0, 255] after applying the gamma correction.
            corrected_array = np.power(image_array / 255.0, gamma) * 255.0

            # Convert the corrected array back to an image
            corrected_img = corrected_array.astype('uint8')
            
            return corrected_img
        #-----------------------------------------------------------------------------------------------------------------------    
        elif id==1:
            # Brightness and contrast stretching
            image_array = np.array(image)
            
            # Brightness
            constant_value = 0
            image_array = image_array + constant_value
            
            
            # Contrast stretching
            u1 = 80
            u2 = 130        

            m = 255 / (u2 - u1)  # m=(y2-y1)/(x2-x1) -> let (u1,0) and (u2,255) -> m=(255-0)/(u2-u1)
                                  # let (u1,0) -> 0=m*u1+b -> b=-m*u1 -> #y=m*x - m*u1         

            thresholded_image = np.where(image_array < u1, 0,  # set values less than u1 to 0
                                        np.where(image_array > u2, 255,  # set values greater than u2 to 255
                                                m * image_array - m * u1)).astype(np.uint8)  # set values between u1 and u2 to (m*image_array - m*u1)

            return thresholded_image                    
#----------------------------------------------------------------------------
'''
for i in range(1, 4):
    path = f'{i}.jpg'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    fixed_image = apply_fix(image, i)
    cv2.imwrite(f'{i}_fixed.jpg', fixed_image)
'''
for i in range(1, 4):
    path = f'{i}.jpg'
    if i!=3:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
            #Dont convert to gray 
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    fixed_image = apply_fix(image, i)
    cv2.imwrite(f'{i}_fixed.jpg', fixed_image)
