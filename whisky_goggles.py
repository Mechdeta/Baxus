import cv2
import numpy as np
import pytesseract
import requests
from io import BytesIO
from typing import List, Tuple, Dict
import json
import os

class WhiskyGoggles:
    def __init__(self, dataset_path: str):
        """Initialize with path to bottle dataset JSON."""
        self.dataset = self.load_dataset(dataset_path)
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
    
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """Load BAXUS dataset of 500 bottles, filtering invalid URLs."""
        with open(dataset_path, 'r') as f:
            data = json.load(f)
            return [entry for entry in data if isinstance(entry.get('image_path'), str) and entry.get('image_path') and not entry.get('image_path').isspace()]
    
    def download_image(self, url: str) -> np.ndarray:
        """Download image from URL and cache it with a valid extension."""
        cache_dir = "image_cache"
        os.makedirs(cache_dir, exist_ok=True)
        # Ensure filename has .jpg extension
        filename = url.split('/')[-1]
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            filename += '.jpg'
        cache_path = os.path.join(cache_dir, filename)
        
        if os.path.exists(cache_path):
            img = cv2.imread(cache_path)
            if img is not None:
                return img
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to decode image from {url}")
            cv2.imwrite(cache_path, img)
            return img
        except Exception as e:
            print(f"Error downloading image from {url}: {e}")
            return None
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for feature detection and OCR."""
        if img is None:
            raise ValueError("Invalid image provided")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        try:
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
        except:
            _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        return cv2.fastNlMeansDenoising(thresh)
    
    def preprocess_local_image(self, image_path: str) -> np.ndarray:
        """Preprocess local image file."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return self.preprocess_image(img)
    
    def extract_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract SIFT features from preprocessed image."""
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        if descriptors is None:
            print("Warning: No descriptors extracted")
            return [], None
        if descriptors.shape[0] > 10000:
            descriptors = descriptors[:10000]
        return keypoints, descriptors
    
    def extract_text(self, image: np.ndarray) -> str:
        """Extract text from image using Tesseract OCR."""
        config = '--oem 3 --psm 6'
        return pytesseract.image_to_string(image, config=config).strip()
    
    def match_bottle(self, query_keypoints, query_descriptors, query_text) -> List[Dict]:
        """Match bottle against dataset using features and text."""
        matches = []
        for bottle in self.dataset:
            ref_path = bottle.get('image_path')
            if not ref_path or ref_path == 'nan':
                print(f"Skipping {bottle['name']} due to invalid URL")
                continue
            
            ref_img = self.download_image(ref_path)
            if ref_img is None:
                continue
            
            ref_processed = self.preprocess_image(ref_img)
            ref_keypoints, ref_descriptors = self.extract_features(ref_processed)
            print(f"Processing {bottle['name']}: ref_descriptors shape = {ref_descriptors.shape if ref_descriptors is not None else 'None'}")
            
            feature_score = 0
            if query_descriptors is not None and ref_descriptors is not None and query_keypoints:
                try:
                    matches_knn = self.matcher.knnMatch(query_descriptors, ref_descriptors, k=2)
                    good_matches = [m for m, n in matches_knn if m.distance < 0.7 * n.distance]  # Tighten ratio test
                    feature_score = len(good_matches) / len(query_keypoints) if query_keypoints else 0
                    if feature_score < 0.1:  # Minimum threshold
                        feature_score = 0
                except cv2.error as e:
                    print(f"Matching error for {bottle['name']}: {e}")
            
            ref_text = bottle.get('label_text', '')
            text_score = self.calculate_text_similarity(query_text, ref_text)
            # Increase text weight for better relevance
            confidence = 0.5 * feature_score + 0.5 * text_score
            
            if confidence > 0.05:  # Minimum confidence threshold
                matches.append({'bottle': bottle, 'confidence': confidence})
        
        return sorted(matches, key=lambda x: x['confidence'], reverse=True)[:3]
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        return len(set(text1.lower().split()) & set(text2.lower().split())) / len(set(text1.lower().split()) | set(text2.lower().split())) or 0.0
    
    def identify_bottle(self, image_path: str) -> List[Dict]:
        """Main function to identify bottle from image and record price."""
        processed_img = self.preprocess_local_image(image_path)
        keypoints, descriptors = self.extract_features(processed_img)
        text = self.extract_text(processed_img)
        price = input("Enter the price (e.g., 49.99): ") or 0.0
        try:
            price = float(price)
        except ValueError:
            price = 0.0
        matches = self.match_bottle(keypoints, descriptors, text)
        for match in matches:
            match['price'] = price
        return matches

def main():
    """Run the bottle identification system."""
    goggles = WhiskyGoggles('baxus_dataset.json')
    image_path = 'sample_bottle.jpg'
    results = goggles.identify_bottle(image_path)
    print("Top matches with price:")
    for result in results:
        bottle = result['bottle']
        print(f"Bottle: {bottle['name']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Brand: {bottle.get('brand', 'N/A')}")
        print(f"Price: ${result['price']:.2f}")
        print("---")

if __name__ == "__main__":
    main()
