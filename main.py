import cv2
import numpy as np
import mediapipe as mp
import sys
print(sys.executable)
# Initialize MediaPipe components
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

# Setup video capture
input_video_path = 'C:\\Users\\glab\\Desktop\\code\\Remove-Background-from-Video-using-TensorFlow\\videos\\man.mp4'
output_video_path = 'C:\\Users\\glab\\Desktop\\code\\Remove-Background-from-Video-using-TensorFlow\\videos\\man2.mp4'
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize the selfie segmentation model
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform segmentation
        results = selfie_segmentation.process(image_rgb)

        # Create a mask and threshold
        condition = results.segmentation_mask > 0.5  # Adjust the threshold if necessary

        # Create a background (white or any other color)
        background = np.ones(frame.shape, dtype=np.uint8) * 255  # White background

        # Use the mask to combine the foreground and background
        output_frame = np.where(condition[:, :, np.newaxis], frame, background)

        # Write the frame to the output video
        out.write(output_frame)

        # Optional: Display the output
        cv2.imshow('Output', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()