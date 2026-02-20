import cv2
import numpy as np
from ultralytics import YOLO

print("Loading YOLO model...")
model = YOLO('yolo11n-seg.pt')
print("✅ Model loaded!")

# Global variables
selected_box = None
blur_intensity = 25
tracking = False

def create_mask(frame, box):
    """Create mask for selected object"""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = map(int, box)
    
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    width = max(10, (x2 - x1) // 2)
    height = max(10, (y2 - y1) // 2)
    
    cv2.ellipse(mask, (center_x, center_y), (width, height), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 11)
    return mask

def apply_blur(frame, mask, intensity):
    """Apply blur to background"""
    blur_size = intensity if intensity % 2 == 1 else intensity + 1
    blurred = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
    
    mask_norm = mask.astype(float) / 255.0
    mask_3ch = np.stack([mask_norm] * 3, axis=2)
    
    result = (frame * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
    return result

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks"""
    global selected_box, tracking
    
    if event == cv2.EVENT_LBUTTONDOWN:
        frame, detections = param
        
        # Find clicked object
        for det in detections:
            x1, y1, x2, y2 = det
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_box = det
                tracking = True
                print(f"✅ Tracking object at ({x1}, {y1}, {x2}, {y2})")
                return

def change_blur(val):
    """Blur intensity trackbar callback"""
    global blur_intensity
    blur_intensity = max(5, val if val % 2 == 1 else val + 1)

def main():
    global selected_box, tracking, blur_intensity
    
    print("\n" + "="*60)
    print("🎯 SMART FOCUS AI - WORKING VERSION")
    print("="*60)
    print("\n📋 Instructions:")
    print("1. Choose: Press 'V' for video or 'W' for webcam")
    print("2. Click on any object to track it")
    print("3. Use slider to adjust blur intensity")
    print("4. Press 'R' to reset tracking")
    print("5. Press 'Q' to quit")
    print("="*60 + "\n")
    
    # Wait for user choice
    print("Press 'V' for Video or 'W' for Webcam...")
    choice = input("Your choice (v/w): ").lower()
    
    if choice == 'v':
        video_path = input("Enter video path (or press Enter for sample): ").strip()
        if not video_path:
            video_path = 'sample.mp4'
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        print(f"✅ Video loaded: {video_path}")
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam!")
            return
        print("✅ Webcam started")
    
    # Create window
    cv2.namedWindow('Smart Focus AI')
    cv2.createTrackbar('Blur', 'Smart Focus AI', 25, 51, change_blur)
    
    print("\n🎬 Processing started! Click on objects to track them.\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            if choice == 'v':
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break
        
        # Run detection
        results = model(frame, verbose=False)[0]
        
        # Get detections
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append([x1, y1, x2, y2])
            
            # Draw all detections in blue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Apply blur if tracking
        if tracking and selected_box is not None:
            # Check if selected box still exists
            found = False
            for det in detections:
                if abs(det[0] - selected_box[0]) < 50 and abs(det[1] - selected_box[1]) < 50:
                    selected_box = det
                    found = True
                    break
            
            if found:
                # Create mask and apply blur
                mask = create_mask(frame, selected_box)
                frame = apply_blur(frame, mask, blur_intensity)
                
                # Highlight selected object in green
                x1, y1, x2, y2 = selected_box
                cv2.rectangle(frame, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 0), 4)
                cv2.putText(frame, "TRACKING", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                tracking = False
                selected_box = None
        
        # Add instructions
        cv2.putText(frame, "Click object to track | R=Reset | Q=Quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if tracking:
            cv2.putText(frame, f"Blur: {blur_intensity}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Set mouse callback
        cv2.setMouseCallback('Smart Focus AI', mouse_callback, (frame.copy(), detections))
        
        # Show frame
        cv2.imshow('Smart Focus AI', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            selected_box = None
            tracking = False
            print("🔄 Tracking reset")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Application closed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
