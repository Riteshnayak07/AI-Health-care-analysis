"""
PulseAI - Computer Vision Module
Analyzes medical images (skin lesions, rashes, eye photos, scan prints)
Performs image quality grading, erythema / rash redness index, and edge enhancement.
"""
import cv2
import numpy as np
import base64

def analyze_image(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read the uploaded image. Please ensure it is a valid JPG or PNG file.")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Quality metrics
    brightness = float(gray.mean())
    contrast = float(gray.std())
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # 2. Erythema / Redness Index (clinical indicator for skin inflammation/rash)
    # OpenCV uses BGR ordering
    b_chan, g_chan, r_chan = cv2.split(image.astype(np.float32))
    redness_ratio = float(np.mean(r_chan / (g_chan + b_chan + 1e-5)))
    
    # 3. Edge density via Canny
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean() * 100)
    
    # Quality grading logic
    warnings = []
    if laplacian_var < 35:
        warnings.append("Low image sharpness (motion blur detected)")
    if brightness < 45:
        warnings.append("Underexposed / too dark")
    elif brightness > 220:
        warnings.append("Overexposed / glare detected")
    if contrast < 25:
        warnings.append("Low visual contrast")

    if not warnings:
        quality = "High Diagnostic Quality"
        quality_score = min(98, round(70 + (laplacian_var / 20.0), 1))
    elif len(warnings) == 1:
        quality = "Acceptable with warnings: " + warnings[0]
        quality_score = 65.0
    else:
        quality = "Sub-optimal: " + ", ".join(warnings)
        quality_score = 42.0

    # Redness Assessment
    if redness_ratio > 0.85:
        erythema_level = "Elevated Erythema / Significant Redness (Potential rash/inflammation)"
    elif redness_ratio > 0.65:
        erythema_level = "Moderate vascular / skin tone coloration"
    else:
        erythema_level = "Normal / Low redness index"

    # Generate edge/contrast enhanced preview thumbnail for visual feedback
    # Combine original image with colored edge overlay
    color_edges = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image, 0.7, color_edges, 0.3, 0)
    
    # Resize thumbnail if too large to keep base64 fast
    max_dim = 600
    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        blended = cv2.resize(blended, (int(width * scale), int(height * scale)))
    
    _, buffer = cv2.imencode(".jpg", blended, [cv2.IMWRITE_JPEG_QUALITY, 85])
    encoded_preview = base64.b64encode(buffer).decode("utf-8")

    result = "Image Processed Successfully"
    confidence = round(quality_score, 1)

    details = {
        "resolution": f"{width} × {height} px",
        "aspect_ratio": f"{round(width/height, 2)}:1",
        "mean_brightness": round(brightness, 1),
        "contrast_std": round(contrast, 1),
        "sharpness_laplacian": round(laplacian_var, 1),
        "redness_index": round(redness_ratio, 3),
        "edge_density_pct": round(edge_density, 1),
        "erythema_assessment": erythema_level,
        "quality_assessment": quality,
        "preview_data_url": f"data:image/jpeg;base64,{encoded_preview}"
    }
    return result, confidence, details
