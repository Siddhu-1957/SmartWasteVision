# Smart Waste Vision ♻️

## 1. Problem Statement
Waste segregation is important for recycling and environmental protection. Manually identifying different types of waste can be time-consuming.

## 2. Proposed Solution
Smart Waste Vision is a computer vision application that allows users to upload waste images and view basic image processing results. AI-based waste classification is an intended feature.

## 3. Technical Approach
The application uses image upload and computer vision techniques to process images and explore automated waste identification.

## 4. Technologies Used
- Python
- Streamlit
- OpenCV
- NumPy
- Pillow
- Hugging Face Transformers (for the attempted AI classification feature)

## 5. Implementation
1. The user uploads an image.
2. The application displays the original image.
3. OpenCV converts the image to grayscale.
4. Canny edge detection highlights image boundaries.
5. An AI classification feature is being integrated to predict waste categories.

## 6. Results
The application supports image uploading, grayscale conversion and edge detection. AI classification is still under testing, and its predictions require further validation.

## 7. Limitations
- Classification accuracy has not been independently evaluated.
- The AI model requires an internet connection for its initial download.
- More waste images and testing are required.

## 8. Future Improvements
- Improve waste classification accuracy.
- Add recyclable and non-recyclable waste guidance.
- Test the application with a larger image dataset.
- Improve the user interface and deployment.
