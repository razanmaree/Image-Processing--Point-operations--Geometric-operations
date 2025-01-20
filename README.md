# Image-Processing--Point-operations--Geometric-operations
## ***Question 1 – Point operations :***
Inside the q1 directory, you are given three broken images and a skeleton of a python script file called imageFix.py.
For each of those images:

a) Choose which correction is the best one for the image.
- Brightness and contrast stretching.
- Gamma correction.
- Histogram equalization.

b) Apply the chosen correction for the image, attach the fixed image along with the parameters used.


## ***Question 2 – Puzzle solving with geometric operations :***
We’re given pieces of a puzzle, and we want to solve the puzzle by “stitching” the pieces together. For example:

![image](https://github.com/user-attachments/assets/7d8ef13e-ced9-4b59-8424-31ce8bdfebc1)

However, it's not that simple since some mischievous entity applied affine or projective transformations to our puzzle pieces. Therefore, we need to find the inverse-transformations for those pieces so we can put them back together.
There are 3 different puzzles, each in its own folder inside the ‘puzzles’ folder, and for each of them you are given the following:
- A folder (‘pieces’) containing all the transformed puzzle pieces.
- A matches.txt file containing pixel coordinate matches between pairs of images. Each pair consists of 3 or 4 couples of matches, depending on whether the transform is affine or projective. The matches are between the coordinates of a point in a puzzle piece and its appropriate place in the entire puzzle canvas.

Assume that we always start from the first image and aim to stitch the other image to the first one. Therefore, the matches are between the pairs (1st,2nd), (1st,3rd), etc. 
