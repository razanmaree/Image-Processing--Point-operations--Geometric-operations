import cv2
import numpy as np
import os
import shutil
import sys


# matches is of (3|4 X 2 X 2) size. Each row is a match - pair of (kp1, kp2) where kpi = (x, y)
def get_transform(matches, is_affine):
    src_points, dst_points = matches[:, 0], matches[:, 1]

    # Convert points to float32
    src_points = src_points.astype(np.float32)
    dst_points = dst_points.astype(np.float32)
    
    #Calculate the transformation from 1st (src_points)to xth (dst_points) image       
    if is_affine:
        T = cv2.getAffineTransform(src_points, dst_points)
    else:
        T = cv2.getPerspectiveTransform(src_points, dst_points)
    
    return T

def stitch(img1, img2):
    height, width, _ = img1.shape

    # Iterate through each row (y-coordinate)
    for y in range(height):
        # Iterate through each column (x-coordinate)
        for x in range(width):
            #If the color (pixel value at (x, y)) is not black, then replace it -> adding img2 to img1
            if any(channel != 0 for channel in img2[y, x]): 
                img1[y,x]=img2[y, x]
    return img1


# Output size is (w, h)
def inverse_transform_target_image(target_img, original_transform, output_size,is_affine):
    #Added is_affine flag as a parameter to check if affine or projective
    #perform the inverse-transform needed to bring xth to its proper place in 1st’s canvas
    if is_affine:
        inverse_transformed_img = cv2.warpAffine(target_img, original_transform, (output_size[1], output_size[0]), flags=cv2.WARP_INVERSE_MAP)
    else:
        inverse_transformed_img = cv2.warpPerspective(target_img, original_transform, (output_size[1], output_size[0]), flags=cv2.WARP_INVERSE_MAP)
    return inverse_transformed_img


# returns a list of pieces file names
def prepare_puzzle(puzzle_dir):
    edited = os.path.join(puzzle_dir, 'abs_pieces')
    if os.path.exists(edited):
        shutil.rmtree(edited)
    os.mkdir(edited)

    affine = 4 - int("affine" in puzzle_dir)

    matches_data = os.path.join(puzzle_dir, 'matches.txt')

    if not os.path.isfile(matches_data) or os.path.getsize(matches_data) == 0:
        print(f'Error: matches.txt is empty or not found for {puzzle_dir}')
        sys.exit(1)

    n_images = len(os.listdir(os.path.join(puzzle_dir, 'pieces')))

    matches = np.loadtxt(matches_data, dtype=np.int64).reshape(n_images - 1, affine, 2, 2)

    return matches, affine == 3, n_images

if __name__ == '__main__':
    lst = ['puzzle_affine_1', 'puzzle_affine_2', 'puzzle_homography_1']
    #lst = ['puzzle_affine_1']

    for puzzle_dir in lst:
        #print(f'Starting {puzzle_dir}')

        puzzle = os.path.join('puzzles', puzzle_dir)

        pieces_pth = os.path.join(puzzle, 'pieces')
        edited = os.path.join(puzzle, 'abs_pieces')        

        matches, is_affine, n_images = prepare_puzzle(puzzle)

        canvas = cv2.imread(os.path.join(pieces_pth, 'piece_1.jpg'))

        # Save the image to abs_pieces
        image_file_path = os.path.join(edited, f'piece_1_absolute.jpg')
        cv2.imwrite(image_file_path, canvas)

        #Go throw all the pieces, applay transform, and then stitch it to the final canvas
        for i in range(2, n_images + 1):
            #Get the matches and applay get_transform
            matches_pair = matches[i - 2]
            transform_matrix = get_transform(matches_pair, is_affine)

            #Applay inverse_transform_target_image using transform_matrix that we calculated    
            target_img = cv2.imread(os.path.join(pieces_pth, f'piece_{i}.jpg'))
            target_img_inverse_transformed = inverse_transform_target_image(target_img, transform_matrix, canvas.shape[:2],is_affine)

            # Save the image to abs_pieces
            image_file_path = os.path.join(edited, f'piece_{i}_absolute.jpg')
            cv2.imwrite(image_file_path, target_img_inverse_transformed)

            #Stitch the transformed image to the final canvas
            canvas = stitch(canvas, target_img_inverse_transformed)

        sol_file = 'solution.jpg'
        cv2.imwrite(os.path.join(puzzle, sol_file), canvas)
