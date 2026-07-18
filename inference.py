"""Inference script for multimodal biometric system."""

import os
import torch
import cv2
import numpy as np
from PIL import Image
import argparse

from config import Config
from models.ear_model import EarCNN
from models.face_model import FaceCNN
from models.fusion import MultimodalBiometricSystem
from utils.preprocessing import EarDetector, FaceDetector, preprocess_image, enhance_ear_image
from utils.dataset import get_transforms


class BiometricInference:
    """Inference class for multimodal biometric recognition."""
    
    def __init__(
        self,
        ear_model_path: str,
        face_model_path: str,
        num_classes: int,
        device: str = 'cpu'
    ):
        """
        Initialize inference system.
        
        Args:
            ear_model_path: Path to trained ear model
            face_model_path: Path to trained face model
            num_classes: Number of person identities
            device: Device to run inference on
        """
        self.device = torch.device(device)
        self.num_classes = num_classes
        
        # Initialize models
        self.ear_model = EarCNN(
            num_classes=num_classes,
            embedding_dim=Config.EMBEDDING_DIM
        ).to(self.device)
        
        self.face_model = FaceCNN(
            num_classes=num_classes,
            embedding_dim=Config.EMBEDDING_DIM
        ).to(self.device)
        
        # Load weights
        self._load_model_weights(self.ear_model, ear_model_path)
        self._load_model_weights(self.face_model, face_model_path)
        
        # Set to evaluation mode
        self.ear_model.eval()
        self.face_model.eval()
        
        # Initialize multimodal system
        self.multimodal_system = MultimodalBiometricSystem(
            ear_model=self.ear_model,
            face_model=self.face_model,
            fusion_method='score',
            ear_weight=Config.EAR_WEIGHT,
            face_weight=Config.FACE_WEIGHT
        ).to(self.device)
        
        # Initialize detectors
        self.ear_detector = EarDetector()
        self.face_detector = FaceDetector()
        
        # Get transforms
        self.transform = get_transforms(is_training=False)
    
    def _load_model_weights(self, model: torch.nn.Module, checkpoint_path: str):
        """Load model weights from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model from {checkpoint_path}')
    
    def preprocess_ear(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess ear image."""
        # Detect and crop ear
        ear_region = self.ear_detector.detect_ear(image)
        if ear_region is None:
            raise ValueError('Could not detect ear in image')
        
        # Enhance
        ear_region = enhance_ear_image(ear_region)
        
        # Resize
        ear_region = cv2.resize(ear_region, Config.EAR_IMAGE_SIZE)
        
        # Apply transforms
        ear_tensor = self.transform(image=ear_region)['image']
        
        return ear_tensor.unsqueeze(0)  # Add batch dimension
    
    def preprocess_face(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess face image."""
        # Detect and crop face
        face_region = self.face_detector.detect_face(image)
        if face_region is None:
            raise ValueError('Could not detect face in image')
        
        # Resize
        face_region = cv2.resize(face_region, Config.FACE_IMAGE_SIZE)
        
        # Apply transforms
        face_tensor = self.transform(image=face_region)['image']
        
        return face_tensor.unsqueeze(0)  # Add batch dimension
    
    def predict(
        self, 
        ear_image_path: str = None,
        face_image_path: str = None,
        profile_image_path: str = None
    ) -> dict:
        """
        Make prediction using ear and face images.
        
        Args:
            ear_image_path: Path to ear image
            face_image_path: Path to face image
            profile_image_path: Path to profile image containing both ear and face
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        if profile_image_path:
            # Load profile image
            profile_img = cv2.imread(profile_image_path)
            profile_img = cv2.cvtColor(profile_img, cv2.COLOR_BGR2RGB)
            
            # Preprocess both modalities
            ear_tensor = self.preprocess_ear(profile_img)
            face_tensor = self.preprocess_face(profile_img)
            
        else:
            # Load separate images
            if ear_image_path:
                ear_img = cv2.imread(ear_image_path)
                ear_img = cv2.cvtColor(ear_img, cv2.COLOR_BGR2RGB)
                ear_tensor = self.preprocess_ear(ear_img)
            else:
                raise ValueError('Either profile_image_path or ear_image_path must be provided')
            
            if face_image_path:
                face_img = cv2.imread(face_image_path)
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_tensor = self.preprocess_face(face_img)
            else:
                raise ValueError('Either profile_image_path or face_image_path must be provided')
        
        # Move to device
        ear_tensor = ear_tensor.to(self.device)
        face_tensor = face_tensor.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            # Individual predictions
            ear_output = self.ear_model(ear_tensor)
            face_output = self.face_model(face_tensor)
            
            ear_probs = torch.softmax(ear_output, dim=1)
            face_probs = torch.softmax(face_output, dim=1)
            
            ear_conf, ear_pred = torch.max(ear_probs, dim=1)
            face_conf, face_pred = torch.max(face_probs, dim=1)
            
            # Multimodal prediction
            multimodal_pred, multimodal_conf = self.multimodal_system.predict(
                ear_tensor, face_tensor
            )
        
        results = {
            'ear_prediction': ear_pred.item(),
            'ear_confidence': ear_conf.item(),
            'face_prediction': face_pred.item(),
            'face_confidence': face_conf.item(),
            'multimodal_prediction': multimodal_pred.item(),
            'multimodal_confidence': multimodal_conf.item()
        }
        
        return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Multimodal Biometric Inference')
    parser.add_argument('--profile', type=str, help='Path to profile image')
    parser.add_argument('--ear', type=str, help='Path to ear image')
    parser.add_argument('--face', type=str, help='Path to face image')
    parser.add_argument('--ear_model', type=str, default='checkpoints/ear_model_best.pth',
                        help='Path to ear model checkpoint')
    parser.add_argument('--face_model', type=str, default='checkpoints/face_model_best.pth',
                        help='Path to face model checkpoint')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='Number of person identities')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Initialize inference system
    inference = BiometricInference(
        ear_model_path=args.ear_model,
        face_model_path=args.face_model,
        num_classes=args.num_classes,
        device=args.device
    )
    
    # Make prediction
    try:
        results = inference.predict(
            ear_image_path=args.ear,
            face_image_path=args.face,
            profile_image_path=args.profile
        )
        
        print('\n' + '='*60)
        print('BIOMETRIC RECOGNITION RESULTS')
        print('='*60)
        print(f'Ear Model:')
        print(f'  Prediction: Person {results["ear_prediction"]}')
        print(f'  Confidence: {results["ear_confidence"]:.4f}')
        print()
        print(f'Face Model:')
        print(f'  Prediction: Person {results["face_prediction"]}')
        print(f'  Confidence: {results["face_confidence"]:.4f}')
        print()
        print(f'Multimodal System:')
        print(f'  Prediction: Person {results["multimodal_prediction"]}')
        print(f'  Confidence: {results["multimodal_confidence"]:.4f}')
        print('='*60)
        
    except Exception as e:
        print(f'Error during inference: {str(e)}')
        return


if __name__ == '__main__':
    main()
