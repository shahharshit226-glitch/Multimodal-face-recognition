"""Utility script to prepare dataset from profile images."""

import os
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path

from utils.preprocessing import EarDetector, FaceDetector


def extract_ear_face_from_profile(
    profile_dir: str,
    output_ear_dir: str,
    output_face_dir: str
):
    """
    Extract ear and face regions from profile images.
    
    Args:
        profile_dir: Directory containing profile images organized by person ID
        output_ear_dir: Output directory for extracted ear images
        output_face_dir: Output directory for extracted face images
    """
    # Initialize detectors
    ear_detector = EarDetector()
    face_detector = FaceDetector()
    
    # Get list of person directories
    person_dirs = [d for d in os.listdir(profile_dir) 
                   if os.path.isdir(os.path.join(profile_dir, d))]
    
    print(f'Found {len(person_dirs)} person directories')
    
    total_processed = 0
    total_failed = 0
    
    for person_id in tqdm(person_dirs, desc='Processing persons'):
        person_profile_dir = os.path.join(profile_dir, person_id)
        person_ear_dir = os.path.join(output_ear_dir, person_id)
        person_face_dir = os.path.join(output_face_dir, person_id)
        
        # Create output directories
        os.makedirs(person_ear_dir, exist_ok=True)
        os.makedirs(person_face_dir, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(person_profile_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in image_files:
            img_path = os.path.join(person_profile_dir, img_file)
            
            try:
                # Load image
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Extract ear
                ear_region = ear_detector.detect_ear(img_rgb)
                if ear_region is not None:
                    ear_output_path = os.path.join(
                        person_ear_dir, 
                        f"ear_{Path(img_file).stem}.jpg"
                    )
                    ear_bgr = cv2.cvtColor(ear_region, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(ear_output_path, ear_bgr)
                else:
                    print(f'Warning: Could not detect ear in {img_path}')
                    total_failed += 1
                    continue
                
                # Extract face
                face_region = face_detector.detect_face(img_rgb)
                if face_region is not None:
                    face_output_path = os.path.join(
                        person_face_dir,
                        f"face_{Path(img_file).stem}.jpg"
                    )
                    face_bgr = cv2.cvtColor(face_region, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(face_output_path, face_bgr)
                else:
                    print(f'Warning: Could not detect face in {img_path}')
                    total_failed += 1
                    continue
                
                total_processed += 1
                
            except Exception as e:
                print(f'Error processing {img_path}: {str(e)}')
                total_failed += 1
    
    print(f'\nProcessing complete!')
    print(f'Successfully processed: {total_processed} images')
    print(f'Failed: {total_failed} images')
    print(f'Ear images saved to: {output_ear_dir}')
    print(f'Face images saved to: {output_face_dir}')


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Extract ear and face regions from profile images'
    )
    parser.add_argument(
        '--profile_dir',
        type=str,
        required=True,
        help='Directory containing profile images organized by person ID'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/raw',
        help='Base output directory (default: data/raw)'
    )
    
    args = parser.parse_args()
    
    # Set output directories
    output_ear_dir = os.path.join(args.output_dir, 'ear')
    output_face_dir = os.path.join(args.output_dir, 'face')
    
    # Create output directories
    os.makedirs(output_ear_dir, exist_ok=True)
    os.makedirs(output_face_dir, exist_ok=True)
    
    # Process images
    extract_ear_face_from_profile(
        args.profile_dir,
        output_ear_dir,
        output_face_dir
    )


if __name__ == '__main__':
    main()
