- Download the dataset from [here](https://github.com/udacity/P1_Facial_Keypoints/tree/master)

One can also use face alignment library to fetch facial keypoint landmarks in 2D or 3D:
``` bash
import face_alignemnt, cv2
im = cv2.imread('image.jpg')
#For 2D keypoints
fa2 = face_alignment.FaceAlignment(face_alignment.LandmarksType.Two_D, flip_input = False, device = 'cpu')
preds2 = fa2.get_landmarks(input)[0]
#For 3D keypoints
fa3 = face_alignment.FaceAlignment(face_alignment.LandmarksType.Three_D, flip_input = False, device = 'cpu')
preds3 = fa3.get_landmarks(input)[0]
```
