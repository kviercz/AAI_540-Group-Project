# preprocessing.py

import os
import cv2
import numpy as np
import boto3
import glob

def main():
    # 1. Define local input & output paths
    input_dir = "/opt/ml/processing/input"
    output_dir = "/opt/ml/processing/output"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Gather all files in the input directory
    #    (SageMaker will have downloaded everything from s3://my-bucket/raw/)
    image_files = glob.glob(os.path.join(input_dir, "*.*"))  # match any extension
    if not image_files:
        print("No files found in input directory.")
        return

    # 3. Process each file
    for image_file in image_files:
        print(f"Processing file: {image_file}")
        image = cv2.imread(image_file)
        if image is None:
            print(f"Warning: Could not read file {image_file}, skipping.")
            continue

        # Convert to JPG if needed
        _, ext = os.path.splitext(image_file)
        if ext.lower() not in ['.jpg', '.jpeg']:
            # Overwrite or rename as .jpg
            new_file = os.path.splitext(image_file)[0] + ".jpg"
            cv2.imwrite(new_file, image)
            image_file = new_file
            image = cv2.imread(image_file)
            print(f"Converted {ext} to .jpg: {new_file}")

        # Detect faces
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            print(f"No faces detected in {image_file}, skipping.")
            continue

        # 4. Process & save each face
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        for idx, (x, y, w, h) in enumerate(faces):
            face_roi = image[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (48, 48))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            face_array = face_gray.astype("float32") / 255.0

            # Convert to 8-bit for saving
            face_uint8 = (face_array * 255).astype("uint8")
            out_face_file = os.path.join(output_dir, f"{base_name}_face_{idx}.png")
            cv2.imwrite(out_face_file, face_uint8)
            print(f"Saved processed face to {out_face_file}")

if __name__ == "__main__":
    main()
