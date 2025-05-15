import time
import numpy as np
import cv2 as cv
from pibot_client import PiBot

# --- Constants ---
AREA_THRESHOLD = 100  # Minimum blob area to be considered a fire
WAIT_TIME = 5         # Seconds to stop at each fire

# --- Simulated global positions (you can calibrate these better)
def estimate_position(cx, cy, frame_width):
    """Estimate fire position based on image x-coordinate"""
    # Divide image into 3 sectors: left, center, right
    if cx < frame_width / 3:
        return "left"
    elif cx > 2 * frame_width / 3:
        return "right"
    else:
        return "center"

def colour_masks(hsv):
    r1_lo, r1_hi = np.array([0, 50, 50]),   np.array([10, 255, 255])
    r2_lo, r2_hi = np.array([170, 50, 50]), np.array([180, 255, 255])
    red = cv.inRange(hsv, r1_lo, r1_hi) | cv.inRange(hsv, r2_lo, r2_hi)
    return red

def detect_fire(mask):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < AREA_THRESHOLD:
            continue
        x, y, w, h = cv.boundingRect(cnt)
        blobs.append({
            'area': area,
            'bbox': (x, y, w, h),
            'centroid': (x + w / 2, y + h / 2)
        })
    return sorted(blobs, key=lambda b: b['area'], reverse=True)

def go_to_beacon(pibot, position_label):
    """Stub: navigate to fixed beacon position"""
    print(f"🧭 Navigating to beacon at {position_label}...")
    # Here you would implement actual motion logic (e.g., turn and drive)
    time.sleep(2)  # simulate travel
    pibot.setVelocity(0, 0)

def main(pibot: PiBot):
    print("🔥 Firefighter Robot Activated")
    time.sleep(1)

    # --- INITIAL SCAN ---
    frame = pibot.getImage()
    if frame is None:
        print("❌ Camera error.")
        return

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    red_mask = colour_masks(hsv)
    red_blobs = detect_fire(red_mask)

    if not red_blobs:
        print("⚠️ No fires detected.")
        return

    # Estimate positions and store sorted list
    fw = frame.shape[1]
    fire_targets = []
    for blob in red_blobs:
        cx, cy = blob['centroid']
        pos_label = estimate_position(cx, cy, fw)
        fire_targets.append({
            'area': blob['area'],
            'position_label': pos_label,
            'bbox': blob['bbox']
        })

    print("\n🧠 Fire detection complete. Visiting in order of intensity:")

    for i, fire in enumerate(fire_targets):
        print(f"  🔥 Fire {i+1} at {fire['position_label']} | Intensity: {int(fire['area'])}")

    # --- VISIT EACH FIRE ---
    for i, fire in enumerate(fire_targets):
        pos = fire['position_label']
        go_to_beacon(pibot, pos)

        x, y, w, h = fire['bbox']
        print(f"\n🚒 Addressing fire {i+1} at {pos} | Intensity: {int(fire['area'])}")
        print(f"🛑 Stopping for {WAIT_TIME} seconds")
        time.sleep(WAIT_TIME)

    print("\n✅ All fires addressed.")

    # Cleanup
    pibot.setVelocity(0, 0)
    pibot.camera.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    pibot = PiBot(ip="172.19.232.164", port=8080, localiser_ip=None)
    main(pibot)
