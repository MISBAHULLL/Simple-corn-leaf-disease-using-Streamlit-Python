"""
Script to crop visualization images to remove code snippets and UI elements.
Crops images to show only the visualization content.
"""

from PIL import Image
import os

def crop_image(input_path, output_path, crop_box=None, crop_percent=None):
    """
    Crop an image.
    
    Args:
        input_path: Path to input image
        output_path: Path to save cropped image
        crop_box: Tuple (left, top, right, bottom) in pixels
        crop_percent: Tuple (left%, top%, right%, bottom%) as decimals
    """
    img = Image.open(input_path)
    width, height = img.size
    
    if crop_percent:
        left = int(width * crop_percent[0])
        top = int(height * crop_percent[1])
        right = int(width * crop_percent[2])
        bottom = int(height * crop_percent[3])
        crop_box = (left, top, right, bottom)
    
    if crop_box:
        cropped = img.crop(crop_box)
        cropped.save(output_path)
        print(f"Cropped: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
        return True
    return False

def main():
    vis_dir = os.path.join(os.path.dirname(__file__), "visualisasi")
    cropped_dir = os.path.join(vis_dir, "cropped")
    
    # Create cropped directory if not exists
    os.makedirs(cropped_dir, exist_ok=True)
    
    # Define crop settings for each image type
    # Format: (original_name, output_name, crop_percent as (left, top, right, bottom))
    
    crop_settings = [
        # Confusion matrices - typically need top portion removed (code output)
        ("evaluasi dan confusion metrik XGboost.png", "confusion_xgboost.png", (0, 0.15, 1, 1)),
        ("evaluasi dan confusion metrik random forest.png", "confusion_rf.png", (0, 0.15, 1, 1)),
        ("evaluasi dan confusion metrik decission tree.png", "confusion_dt.png", (0, 0.15, 1, 1)),
        
        # SHAP plots
        ("plot summary shap.png", "shap_summary.png", (0, 0, 1, 1)),  # Usually clean
        ("shap force.png", "shap_force.png", (0, 0, 1, 1)),
        ("shap interaction value.png", "shap_interaction.png", (0, 0, 1, 1)),
        
        # Model comparison
        ("perbandingan akurasi model.png", "model_comparison.png", (0, 0, 1, 1)),
        
        # Distribution - screenshots may have extra UI
        ("Screenshot 2026-01-02 230604.png", "distribution.png", (0, 0.05, 1, 0.95)),
        ("Screenshot 2026-01-02 230643.png", "sample_images.png", (0, 0.05, 1, 0.95)),
    ]
    
    for original, output, crop_pct in crop_settings:
        input_path = os.path.join(vis_dir, original)
        output_path = os.path.join(cropped_dir, output)
        
        if os.path.exists(input_path):
            crop_image(input_path, output_path, crop_percent=crop_pct)
        else:
            print(f"Not found: {original}")
    
    print(f"\nCropped images saved to: {cropped_dir}")

if __name__ == "__main__":
    main()
