import cv2
from PIL import Image, ImageOps, ImageEnhance
from tensorflow import keras
import numpy

def predict(input_path, model):
    image = Image.open(input_path)
    pixels = [pixel for pixel in image.getdata()]

    print(pixels)

    prediction = model.predict(numpy.array([pixels], dtype=numpy.uint8), verbose=0)
    return 1 if prediction < 0.5 else 2

def process_image(input_path, output_path):
    with Image.open(input_path) as img:
         # Get dimensions
        width, height = img.size
        
        # Calculate dimensions for center crop
        size = min(width, height)
        
        # Calculate coordinates for center crop
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size
        
        # Crop the image to a square from the center
        cropped_img = img.crop((left, top, right, bottom))
        
        # Convert to black and white (grayscale)
        grayscale_img = cropped_img.convert('L')
        
        # Increase contrast
        # enhancer = ImageEnhance.Contrast(grayscale_img)
        # contrast_img = enhancer.enhance(5)

        threshold = 100
        bw_img = grayscale_img.point(lambda x: 0 if x < threshold else 255, '1')

        # Invert the colors
        inverted_img = ImageOps.invert(bw_img)

        # Resize to 28x28
        resized_img = inverted_img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Save the processed image
        resized_img.save(output_path)
        
model = keras.models.load_model("model.keras")

# Initialize the video capture object (0 for default camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    raise IOError("Cannot open webcam")

window_size = 500
window_name = "Preview"

# Create a window named 'Preview'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while(True):
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get dimensions
    height, width = frame.shape[:2]
    
    # Calculate dimensions for center crop
    size = min(width, height)
    
    # Calculate coordinates for center crop
    left = (width - size) // 2
    top = (height - size) // 2
    
    # Crop the frame to a square from the center
    square_frame = frame[top:top+size, left:left+size]
    
    # Resize the square frame if needed
    if size != window_size:
        square_frame = cv2.resize(square_frame, (window_size, window_size))
    
    # Display the square-cropped frame
    cv2.imshow(window_name, square_frame)

    keyPress = cv2.waitKey(1) & 0xFF
    if keyPress == ord('q'):
        break
    elif keyPress == ord('c'):
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            raise IOError("Cannot read frame")

        # Save the captured frame as an image
        cv2.imwrite("captured_image.png", frame)

        process_image("captured_image.png", "processed_image.png")

        prediction = predict("processed_image.png", model)

        print(f"Prediction: {prediction}")

# Release the video capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()