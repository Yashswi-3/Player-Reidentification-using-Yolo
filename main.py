# main.py - Enhanced Player Tracker with ID Stability Fixes

import cv2
import numpy as np
import torch
import time
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import json

class PlayerTracker:
    def __init__(self, model_path, video_path, conf_thres=0.5, iou_thres=0.45):
        """Initialize the enhanced player tracker with ID stability improvements."""
        self.model_path = Path(model_path)
        self.video_path = Path(video_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Load model
        self.model = self._load_model()
        
        # Enhanced tracking parameters for ID stability
        self.tracks = {}                        # Active tracks
        self.disappeared_tracks = {}            # Memory for disappeared players

        self.next_id = 1
        self.max_disappeared = 75               # Extended for better stability
        self.max_distance = 150                 # Maximum distance for track matching
        self.reid_threshold = 0.65              # Threshold for re-identification
        self.memory_duration = 1000             # Keep embeddings for entire video
        
        # Critical ID stability parameters
        self.hungarian_threshold = 0.65         # Raised from 0.4 for stricter matching
        self.feature_smoothing_alpha = 0.7      # For temporal feature smoothing
        self.motion_gate_threshold = 150        # Motion prediction gate
        self.min_feature_similarity = 0.3      # Outlier rejection threshold
        self.track_creation_cooldown = 0.5      # Minimum similarity for new tracks
        self.secondary_iou_threshold = 0.1      # Secondary verification IoU
        self.secondary_feature_threshold = 0.5  # Secondary verification feature
        
        # Motion prediction and velocity tracking
        self.track_velocities = {}              # Store velocity for each track
        self.track_positions_history = {}       # Store position history for smoothing
        
        # Feature extractor for re-identification
        self.feature_extractor = self._init_feature_extractor()
        
        # Video capture
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Statistics tracking
        self.reidentification_count = 0
        self.track_history = {}
        self.id_switches = 0
        self.similarity_logs = []
        
        print(f"Video loaded: {self.width}x{self.height} @ {self.fps}fps, {self.total_frames} frames")
        print(f"Enhanced ID stability parameters loaded")

    def _load_model(self):
        """Load the YOLO model using Ultralytics."""
        try:
            from ultralytics import YOLO
            model = YOLO(str(self.model_path))
            print(f"Model loaded successfully with Ultralytics from {self.model_path}")
            print(f"Model classes: {model.names}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _init_feature_extractor(self):
        """Initialize feature extractor for re-identification."""
        try:
            import torchvision.models as models
            resnet = models.resnet18(pretrained=True)
            # Remove final classification layer
            feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
            feature_extractor.eval()
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            feature_extractor.to(device)
            self.device = device
            
            print(f"Feature extractor loaded successfully on {device}")
            return feature_extractor
            
        except Exception as e:
            print(f"Warning: Could not load feature extractor: {e}")
            print("Using enhanced dummy features for tracking")
            self.device = torch.device('cpu')
            return None

    def smooth_features(self, old_features, new_features, alpha=0.7):
        """Apply temporal smoothing to features using exponential moving average."""
        if old_features is None:
            return new_features
        
        # Check for outlier features (too different from previous)
        similarity = self.cosine_similarity(old_features, new_features)
        if similarity < self.min_feature_similarity:
            print(f"Outlier feature detected (sim: {similarity:.3f}), keeping old features")
            return old_features
        
        # Apply exponential moving average
        smoothed = alpha * old_features + (1 - alpha) * new_features
        # Normalize
        smoothed = smoothed / (np.linalg.norm(smoothed) + 1e-8)
        
        return smoothed.astype(np.float32)

    def predict_track_position(self, track_id, track_data):
        """Predict next position using simple constant velocity model."""
        if track_id not in self.track_velocities:
            return track_data['bbox']
        
        bbox = track_data['bbox']
        velocity = self.track_velocities[track_id]
        
        # Predict center position
        center_x = (bbox[0] + bbox[2]) / 2 + velocity[0]
        center_y = (bbox[1] + bbox[3]) / 2 + velocity[1]
        
        # Keep same size
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        predicted_bbox = [
            center_x - width/2,
            center_y - height/2,
            center_x + width/2,
            center_y + height/2
        ]
        
        return predicted_bbox

    def update_track_velocity(self, track_id, old_bbox, new_bbox):
        """Update velocity estimation for motion prediction."""
        old_center = np.array([(old_bbox[0] + old_bbox[2])/2, (old_bbox[1] + old_bbox[3])/2])
        new_center = np.array([(new_bbox[0] + new_bbox[2])/2, (new_bbox[1] + new_bbox[3])/2])
        
        velocity = new_center - old_center
        
        # Smooth velocity with previous estimate
        if track_id in self.track_velocities:
            old_velocity = self.track_velocities[track_id]
            velocity = 0.7 * old_velocity + 0.3 * velocity
        
        self.track_velocities[track_id] = velocity

    def extract_multi_crop_features(self, frame, bbox):
        """Extract features from multiple crops around the bounding box."""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within frame bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return self.extract_features(frame, bbox)
        
        # Define multiple crop regions
        width = x2 - x1
        height = y2 - y1
        
        crops = [
            # Center crop (original)
            [x1, y1, x2, y2],
            # Slightly expanded crop
            [max(0, x1 - width*0.1), max(0, y1 - height*0.1), 
             min(frame.shape[1], x2 + width*0.1), min(frame.shape[0], y2 + height*0.1)],
            # Upper body focus
            [x1, y1, x2, y1 + height*0.7]
        ]
        
        features_list = []
        for crop_bbox in crops:
            try:
                crop_features = self.extract_features(frame, crop_bbox)
                features_list.append(crop_features)
            except:
                continue
        
        if features_list:
            # Average the features
            avg_features = np.mean(features_list, axis=0)
            # Normalize
            avg_features = avg_features / (np.linalg.norm(avg_features) + 1e-8)
            return avg_features.astype(np.float32)
        else:
            return self.extract_features(frame, bbox)

    def extract_features(self, frame, bbox):
        """Extract comprehensive features from a player crop for re-identification."""
        # Extract multiple types of features
        deep_features = self.extract_deep_features(frame, bbox)
        color_features = self.extract_color_histogram(frame, bbox)
        spatial_features = self.extract_spatial_features(bbox, frame.shape)
        
        # Combine features with weights
        combined_features = np.concatenate([
            deep_features * 0.6,
            color_features * 0.3,
            spatial_features * 0.1
        ])
        
        return combined_features.astype(np.float32)

    def extract_deep_features(self, frame, bbox):
        """Extract deep features using ResNet with enhanced stability."""
        if self.feature_extractor is None:
            # Enhanced dummy features based on bbox characteristics
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / (height + 1e-8)
            
            # Create more sophisticated dummy features
            features = np.array([
                center_x / self.width,
                center_y / self.height,
                width / self.width,
                height / self.height,
                aspect_ratio,
                (width * height) / (self.width * self.height)
            ] + [0.0] * 506)  # Total 512 features
            
            return features.astype(np.float32)
        
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return np.random.rand(512).astype(np.float32)
            
            # Crop player region
            player_crop = frame[y1:y2, x1:x2]
            if player_crop.size == 0:
                return np.random.rand(512).astype(np.float32)
            
            # Resize and normalize
            player_crop = cv2.resize(player_crop, (224, 224))
            player_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor
            tensor = torch.tensor(player_crop).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(self.device)
            
            # Normalize tensor
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
            tensor = (tensor - mean) / std
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(tensor)
                features = features.squeeze().cpu().numpy()
                # Normalize features
                features = features / (np.linalg.norm(features) + 1e-8)
            
            return features.astype(np.float32)
            
        except Exception as e:
            print(f"Error extracting deep features: {e}")
            return np.random.rand(512).astype(np.float32)

    def extract_color_histogram(self, frame, bbox):
        """Extract enhanced color histogram for jersey/appearance matching."""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return np.zeros(100).astype(np.float32)
            
            crop = frame[y1:y2, x1:x2]
            
            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # Focus on upper body for jersey colors (top 70% of bbox)
            upper_crop = hsv[:int(hsv.shape[0] * 0.7), :]
            
            # Calculate histogram for H and S channels
            hist_h = cv2.calcHist([upper_crop], [0], None, [50], [0, 180])
            hist_s = cv2.calcHist([upper_crop], [1], None, [50], [0, 256])
            
            # Combine histograms
            hist = np.concatenate([hist_h.flatten(), hist_s.flatten()])
            hist = hist / (np.sum(hist) + 1e-8)  # Normalize
            
            return hist.astype(np.float32)
            
        except Exception as e:
            print(f"Error extracting color features: {e}")
            return np.zeros(100).astype(np.float32)

    def extract_spatial_features(self, bbox, frame_shape):
        """Extract spatial position features."""
        x1, y1, x2, y2 = bbox
        height, width = frame_shape[:2]
        
        # Normalized position and size features
        center_x = (x1 + x2) / (2 * width)
        center_y = (y1 + y2) / (2 * height)
        bbox_width = (x2 - x1) / width
        bbox_height = (y2 - y1) / height
        aspect_ratio = bbox_width / (bbox_height + 1e-8)
        area = bbox_width * bbox_height
        
        # Position in frame quadrants
        left_half = 1.0 if center_x < 0.5 else 0.0
        top_half = 1.0 if center_y < 0.5 else 0.0
        
        spatial_features = np.array([
            center_x, center_y, bbox_width, bbox_height,
            aspect_ratio, area, left_half, top_half
        ])
        
        return spatial_features.astype(np.float32)

    def cosine_similarity(self, a, b):
        """Calculate cosine similarity between two feature vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def calculate_distance(self, bbox1, bbox2):
        """Calculate Euclidean distance between centers of two bounding boxes."""
        center1 = np.array([(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2])
        center2 = np.array([(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2])
        return np.linalg.norm(center1 - center2)

    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-8)

    def enhanced_similarity_calculation(self, detection, track, track_id):
        """Calculate enhanced similarity with improved weighting and motion gates."""
        bbox, features, conf = detection
        
        # 1. Distance similarity
        distance = self.calculate_distance(bbox, track['bbox'])
        distance_sim = 1 / (1 + distance / self.max_distance)
        
        # 2. IoU similarity (increased weight)
        iou_sim = self.calculate_iou(bbox, track['bbox'])
        
        # 3. Feature similarity with smoothed features
        if features is not None and track['features'] is not None:
            feature_sim = self.cosine_similarity(features, track['features'])
        else:
            feature_sim = 0.5
        
        # 4. Motion gate check
        predicted_bbox = self.predict_track_position(track_id, track)
        motion_distance = self.calculate_distance(bbox, predicted_bbox)
        motion_gate_passed = motion_distance < self.motion_gate_threshold
        
        if not motion_gate_passed:
            # Heavily penalize detections outside motion gate
            return 0.1
        
        # Enhanced weighting: More emphasis on IoU and features for stability
        combined_similarity = (0.2 * distance_sim + 
                             0.3 * iou_sim + 
                             0.5 * feature_sim)
        
        return combined_similarity

    def detect_players(self, frame):
        """Detect players in the frame using YOLO model."""
        try:
            results = self.model(frame, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf.item()
                        cls = int(box.cls.item())
                        class_name = self.model.names[cls]
                        
                        # Filter for players and goalkeepers only
                        if conf > self.conf_thres and class_name in ['player', 'goalkeeper']:
                            bbox = [int(x1), int(y1), int(x2), int(y2)]
                            
                            # Size and aspect ratio filtering to avoid spurious detections
                            width = bbox[2] - bbox[0]
                            height = bbox[3] - bbox[1]
                            aspect_ratio = width / (height + 1e-8)
                            
                            # Filter out too small or oddly shaped detections
                            if width > 20 and height > 40 and 0.2 < aspect_ratio < 2.0:
                                features = self.extract_multi_crop_features(frame, bbox)
                                detections.append((bbox, features, conf))
            
            return detections
            
        except Exception as e:
            print(f"Error in detect_players: {e}")
            return []

    def hungarian_assignment_with_verification(self, similarity_matrix, detections, track_ids):
        """Enhanced Hungarian assignment with secondary verification."""
        if similarity_matrix.size == 0:
            return []
        
        # Log similarity scores for debugging
        for i in range(similarity_matrix.shape[0]):
            scores = similarity_matrix[i, :]
            if len(scores) > 0:
                best_idx = np.argmax(scores)
                second_best_idx = np.argsort(scores)[-2] if len(scores) > 1 else best_idx
                
                best_score = scores[best_idx]
                second_best_score = scores[second_best_idx] if len(scores) > 1 else 0
                
                # Flag potential ambiguity
                if second_best_score > 0.9 * best_score and best_score > 0.5:
                    print(f"Potential ambiguity: best={best_score:.3f}, second={second_best_score:.3f}")
        
        # Apply Hungarian algorithm
        cost_matrix = 1 - similarity_matrix
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        verified_matches = []
        for row, col in zip(row_indices, col_indices):
            similarity = similarity_matrix[row, col]
            
            # Primary threshold check (raised threshold)
            if similarity > self.hungarian_threshold:
                # Secondary verification
                bbox = detections[row][0]
                track = self.tracks[track_ids[col]]
                
                # IoU and feature verification
                iou = self.calculate_iou(bbox, track['bbox'])
                feature_sim = self.cosine_similarity(detections[row][1], track['features'])
                
                # Strict verification: Both IoU and feature similarity must be reasonable
                if iou > self.secondary_iou_threshold or feature_sim > self.secondary_feature_threshold:
                    verified_matches.append((row, col, similarity))
                else:
                    print(f"Match rejected: IoU={iou:.3f}, Feature_sim={feature_sim:.3f}")
        
        return verified_matches

    def should_create_new_track(self, detection, active_tracks, disappeared_tracks):
        """Determine if a new track should be created with enhanced cooldown period."""
        bbox, features, conf = detection
        
        # Check similarity to all existing tracks (active + disappeared)
        max_similarity = 0
        
        # Check active tracks
        for track in active_tracks.values():
            if track['features'] is not None:
                sim = self.cosine_similarity(features, track['features'])
                max_similarity = max(max_similarity, sim)
        
        # Check disappeared tracks
        for track_data in disappeared_tracks.values():
            if track_data['features'] is not None:
                sim = self.cosine_similarity(features, track_data['features'])
                max_similarity = max(max_similarity, sim)
        
        # Only create new track if sufficiently different from all existing
        should_create = max_similarity < self.track_creation_cooldown
        
        if not should_create:
            print(f"Skipped new track creation - max similarity: {max_similarity:.3f}")
        
        return should_create

    def handle_reidentification(self, unmatched_detections, frame_count):
        """Enhanced re-identification with appearance-only backup."""
        reidentified = []
        
        for i, (bbox, features, conf) in enumerate(unmatched_detections):
            best_match_id = None
            best_similarity = 0
            
            # Compare with disappeared tracks
            for old_id, stored_data in self.disappeared_tracks.items():
                # Primary: Combined similarity
                similarity = self.cosine_similarity(features, stored_data['features'])
                
                # Additional spatial consistency check
                spatial_consistency = 1.0
                if 'last_bbox' in stored_data:
                    distance = self.calculate_distance(bbox, stored_data['last_bbox'])
                    spatial_consistency = 1 / (1 + distance / 200)  # Spatial weight
                
                combined_similarity = 0.8 * similarity + 0.2 * spatial_consistency
                
                # Appearance-only backup for edge cases
                if similarity > 0.8 and combined_similarity < self.reid_threshold:
                    print(f"Using appearance-only backup for ID {old_id}")
                    combined_similarity = similarity
                
                if combined_similarity > best_similarity and combined_similarity > self.reid_threshold:
                    best_similarity = combined_similarity
                    best_match_id = old_id
            
            if best_match_id:
                # Reactivate old track
                self.tracks[best_match_id] = {
                    'bbox': bbox,
                    'features': features,
                    'confidence': conf,
                    'disappeared': 0,
                    'last_seen': frame_count,
                    'created': self.disappeared_tracks[best_match_id]['created'],
                    'reidentified': True
                }
                
                # Update track history
                if best_match_id not in self.track_history:
                    self.track_history[best_match_id] = []
                self.track_history[best_match_id].append({
                    'frame': frame_count,
                    'action': 'reidentified',
                    'similarity': best_similarity
                })
                
                del self.disappeared_tracks[best_match_id]
                reidentified.append(i)
                self.reidentification_count += 1
                
                print(f"Re-identified player ID {best_match_id} with similarity {best_similarity:.3f}")
        
        return reidentified

    def update_tracks(self, detections, frame_count):
        """Enhanced track update with all stability improvements."""
        if not detections:
            for track_id in self.tracks:
                self.tracks[track_id]['disappeared'] += 1
            self._handle_disappeared_tracks(frame_count)
            return

        track_ids = list(self.tracks.keys())
        if not track_ids:
            # Create initial tracks with enhanced filtering
            for detection in detections:
                if self.should_create_new_track(detection, {}, self.disappeared_tracks):
                    bbox, features, conf = detection
                    track_id = self.next_id
                    self.tracks[track_id] = {
                        'bbox': bbox,
                        'features': features,
                        'confidence': conf,
                        'disappeared': 0,
                        'last_seen': frame_count,
                        'created': frame_count,
                        'reidentified': False
                    }
                    
                    # Initialize track history
                    self.track_history[track_id] = [{
                        'frame': frame_count,
                        'action': 'created'
                    }]
                    
                    self.next_id += 1
            return

        # Build enhanced similarity matrix
        similarity_matrix = np.zeros((len(detections), len(track_ids)))
        
        for i, detection in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                similarity_matrix[i, j] = self.enhanced_similarity_calculation(detection, track, track_id)

        # Enhanced Hungarian assignment with verification
        matches = self.hungarian_assignment_with_verification(similarity_matrix, detections, track_ids)
        
        matched_detections = set()
        matched_tracks = set()
        
        # Update matched tracks with enhanced feature smoothing
        for det_idx, track_idx, similarity in matches:
            bbox, features, conf = detections[det_idx]
            track_id = track_ids[track_idx]
            old_track = self.tracks[track_id].copy()
            
            # Apply temporal feature smoothing
            smoothed_features = self.smooth_features(
                old_track['features'], 
                features, 
                self.feature_smoothing_alpha
            )
            
            # Update velocity estimation
            self.update_track_velocity(track_id, old_track['bbox'], bbox)
            
            # Update track with smoothed features
            self.tracks[track_id].update({
                'bbox': bbox,
                'features': smoothed_features,  # Use smoothed features
                'confidence': conf,
                'disappeared': 0,
                'last_seen': frame_count
            })
            
            matched_detections.add(det_idx)
            matched_tracks.add(track_idx)

        # Handle unmatched detections with enhanced re-identification
        unmatched_detections = [detections[i] for i in range(len(detections)) 
                               if i not in matched_detections]
        
        if unmatched_detections:
            reidentified_indices = self.handle_reidentification(unmatched_detections, frame_count)
            
            # Create new tracks only for truly new players
            for i, detection in enumerate(unmatched_detections):
                if i not in reidentified_indices:
                    if self.should_create_new_track(detection, self.tracks, self.disappeared_tracks):
                        bbox, features, conf = detection
                        track_id = self.next_id
                        self.tracks[track_id] = {
                            'bbox': bbox,
                            'features': features,
                            'confidence': conf,
                            'disappeared': 0,
                            'last_seen': frame_count,
                            'created': frame_count,
                            'reidentified': False
                        }
                        
                        # Initialize track history
                        self.track_history[track_id] = [{
                            'frame': frame_count,
                            'action': 'created'
                        }]
                        
                        self.next_id += 1

        # Update disappeared counter for unmatched tracks
        for j, track_id in enumerate(track_ids):
            if j not in matched_tracks:
                self.tracks[track_id]['disappeared'] += 1

        # Handle disappeared tracks
        self._handle_disappeared_tracks(frame_count)

    def _handle_disappeared_tracks(self, frame_count):
        """Move disappeared tracks to memory for potential re-identification."""
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            if track['disappeared'] > self.max_disappeared:
                # Move to disappeared tracks memory
                self.disappeared_tracks[track_id] = {
                    'features': track['features'].copy(),
                    'last_bbox': track['bbox'].copy(),
                    'disappeared_frame': frame_count,
                    'created': track['created'],
                    'confidence': track['confidence']
                }
                
                # Update track history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append({
                    'frame': frame_count,
                    'action': 'disappeared'
                })
                
                tracks_to_remove.append(track_id)
        
        # Remove from active tracks
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            # Keep velocity history for potential reactivation
            # Don't delete from track_velocities immediately
        
        # Clean old disappeared tracks (optional memory management)
        old_disappeared = []
        for track_id, data in self.disappeared_tracks.items():
            if frame_count - data['disappeared_frame'] > self.memory_duration:
                old_disappeared.append(track_id)
        
        for track_id in old_disappeared:
            del self.disappeared_tracks[track_id]
            if track_id in self.track_velocities:
                del self.track_velocities[track_id]

    def draw_tracks(self, frame, frame_count):
        """Draw tracking results with enhanced visualization and stability indicators."""
        for track_id, track in self.tracks.items():
            if track['disappeared'] > 0:
                continue
            
            bbox = track['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Color coding: Green for normal, Blue for re-identified, Yellow for stable
            if track.get('reidentified', False):
                color = (255, 0, 0)  # Blue for re-identified
                label_prefix = "RE-ID"
            else:
                color = (0, 255, 0)  # Green for normal tracks
                label_prefix = "ID"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and confidence
            label = f"{label_prefix}: {track_id} ({track['confidence']:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw motion trail for stability visualization
            if track_id in self.track_velocities:
                velocity = self.track_velocities[track_id]
                if np.linalg.norm(velocity) > 5:  # Only show if moving
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    end_point = (int(center[0] - velocity[0] * 3), int(center[1] - velocity[1] * 3))
                    cv2.arrowedLine(frame, center, end_point, color, 2)
        
        # Enhanced frame info with stability metrics
        current_frame = frame_count
        active_tracks = len([t for t in self.tracks.values() if t['disappeared'] == 0])
        disappeared_tracks = len(self.disappeared_tracks)
        
        info_lines = [
            f"Frame: {current_frame}/{self.total_frames}",
            f"Active: {active_tracks} | Memory: {disappeared_tracks}",
            f"Re-IDs: {self.reidentification_count} | Switches: {self.id_switches}",
            f"Threshold: {self.hungarian_threshold:.2f}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

    def calculate_id_stability_metrics(self):
        """Calculate comprehensive ID stability metrics."""
        stability_metrics = {
            'total_tracks_created': self.next_id - 1,
            'successful_reidentifications': self.reidentification_count,
            'id_switches': self.id_switches,
            'tracks_in_memory': len(self.disappeared_tracks),
            'final_active_tracks': len(self.tracks),
            'average_track_lifetime': 0,
            'stability_score': 0
        }
        
        # Calculate average track lifetime
        if self.track_history:
            lifetimes = []
            for track_id, history in self.track_history.items():
                created_frame = history[0]['frame']
                last_frame = history[-1]['frame']
                lifetime = last_frame - created_frame
                lifetimes.append(lifetime)
            
            stability_metrics['average_track_lifetime'] = np.mean(lifetimes)
        
        # Calculate stability score (lower is better)
        total_possible_switches = max(1, self.total_frames * len(self.tracks))
        stability_metrics['stability_score'] = 1 - (self.id_switches / total_possible_switches)
        
        return stability_metrics

    def save_tracking_statistics(self, output_path="output/tracking_stats.json"):
        """Save comprehensive tracking statistics."""
        stability_metrics = self.calculate_id_stability_metrics()
        
        stats = {
            'tracking_parameters': {
                'hungarian_threshold': self.hungarian_threshold,
                'feature_smoothing_alpha': self.feature_smoothing_alpha,
                'motion_gate_threshold': self.motion_gate_threshold,
                'track_creation_cooldown': self.track_creation_cooldown
            },
            'performance_metrics': stability_metrics,
            'track_history': self.track_history,
            'similarity_logs': self.similarity_logs[-100:]  # Last 100 entries
        }
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Enhanced tracking statistics saved to {output_path}")
        print(f"Stability Score: {stability_metrics['stability_score']:.3f}")

    def run(self, save_output=True, output_path="output/stable_tracking_output.mp4"):
        """Run the enhanced player tracking with ID stability improvements."""
        print("Starting enhanced player tracking with ID stability fixes...")
        print(f"Hungarian threshold: {self.hungarian_threshold}")
        print(f"Feature smoothing alpha: {self.feature_smoothing_alpha}")
        print(f"Motion gate threshold: {self.motion_gate_threshold}")
        
        # Initialize video writer if saving output
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                frame_count += 1
                
                # Detect players with enhanced filtering
                detections = self.detect_players(frame)
                
                # Update tracks with all stability improvements
                self.update_tracks(detections, frame_count)
                
                # Draw results with stability visualization
                vis_frame = self.draw_tracks(frame.copy(), frame_count)
                
                # Save frame if needed
                if save_output:
                    out.write(vis_frame)
                
                # Display frame (optional - comment out for faster processing)
                cv2.imshow('Stable Player Tracking', vis_frame)
                
                # Print progress with stability metrics
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    active_tracks = len([t for t in self.tracks.values() if t['disappeared'] == 0])
                    print(f"Processed {frame_count}/{self.total_frames} frames @ {fps:.1f} FPS")
                    print(f"Active: {active_tracks} | Memory: {len(self.disappeared_tracks)} | Re-IDs: {self.reidentification_count} | Switches: {self.id_switches}")
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nTracking interrupted by user")
        except Exception as e:
            print(f"Error during tracking: {e}")
        finally:
            # Cleanup
            self.cap.release()
            if save_output:
                out.release()
            cv2.destroyAllWindows()
            
            elapsed = time.time() - start_time
            avg_fps = frame_count / elapsed if elapsed > 0 else 0
            unique_players = self.next_id - 1
            
            print(f"\n=== ENHANCED TRACKING COMPLETED ===")
            print(f"Processed {frame_count} frames in {elapsed:.2f}s @ {avg_fps:.1f} FPS")
            print(f"Total unique players tracked: {unique_players}")
            print(f"Successful re-identifications: {self.reidentification_count}")
            print(f"ID switches: {self.id_switches}")
            print(f"Final active tracks: {len(self.tracks)}")
            print(f"Tracks in memory: {len(self.disappeared_tracks)}")
            
            if save_output:
                print(f"Output saved to: {output_path}")
            
            # Save comprehensive statistics
            self.save_tracking_statistics()


def main():
    """Main function to run enhanced stable player tracking."""
    # Configuration
    model_path = "models/best.pt"
    video_path = "data/15sec_input_720p.mp4"
    
    # Check if files exist
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Initialize and run tracker with enhanced stability parameters
    tracker = PlayerTracker(
        model_path=model_path,
        video_path=video_path,
        conf_thres=0.5,  # Confidence threshold
        iou_thres=0.45   # IoU threshold for NMS
    )
    
    # Run enhanced stable tracking
    tracker.run(save_output=True, output_path="output/stable_tracking_output.mp4")


if __name__ == "__main__":
    main()
