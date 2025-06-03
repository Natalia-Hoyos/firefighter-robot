import time
import numpy as np
import cv2 as cv
from pibot_client import PiBot
import math 

# --- Constants ---
AREA_THRESHOLD = 100  # Minimum blob area to be considered a fire
STARTING_POSITION = [1.838, 1.712, -140]

# Known positions of the beacons (x, y) in meters
KNOWN_BEACON_POSITIONS = {
    "left": (0.708, 0.362),    
    "center": (0.496, 0.68),  
    "right": (0.31, 0.992)   
}

class SimplePiBotController:
    """odometry controller integrated into the main script"""
    def __init__(self, pibot, target_point, wheel_base=0.15):
        self.WHEEL_RADIUS   = 0.0325
        self.TICKS_PER_REV  = 380
        self.pibot          = pibot
        self.wheel_base     = wheel_base

        x, y, deg = STARTING_POSITION
        self.current_pose = [x, y, math.radians(deg)] 
        self.target_point = target_point
        self.trajectory   = [self.current_pose.copy()]

        # PID parameters
        self.Kp = 1.1
        self.Ki = 0.005
        self.Kd = 0.3
        self.integral_yaw   = 0.0
        self.prev_yaw_error = 0.0
        self.max_rot_cmd    = 1.5
        self.position_threshold = 0.05
        self.time_limit        = 30.0

        # Encoder state
        self.prev_left_ticks, self.prev_right_ticks = self.pibot.getEncoders()

    def reset_odometry(self):
        """Reset to the global starting position"""
        x, y, deg = STARTING_POSITION
        self.current_pose = [x, y, math.radians(deg)]
        self.trajectory = [self.current_pose.copy()]
        print(f"🔄 Odometry reset to: {STARTING_POSITION}")


    def motor_unit_from_velocity(self, v):
        if v > 0:
            return (v + 0.0033) / 0.005515
        elif v < 0:
            return (v - 0.0033) / 0.005515
        return 0

    def set_wheel_speeds_forward(self, forward_v, rot_omega):
        heading_scale = max(0.3, 1 - abs(rot_omega)/self.max_rot_cmd)  # More aggressive turning
        forward_v *= heading_scale
        rot_omega = np.clip(rot_omega, -self.max_rot_cmd, self.max_rot_cmd)

        v_left = forward_v - 0.5 * self.wheel_base * rot_omega
        v_right = forward_v + 0.5 * self.wheel_base * rot_omega

        left_cmd = int(np.clip(self.motor_unit_from_velocity(v_left), -100, 100))
        right_cmd = int(np.clip(self.motor_unit_from_velocity(v_right), -100, 100))

        MIN_CMD = 15
        if 0 < abs(left_cmd) < MIN_CMD: left_cmd = MIN_CMD * np.sign(left_cmd)
        if 0 < abs(right_cmd) < MIN_CMD: right_cmd = MIN_CMD * np.sign(right_cmd)

        self.pibot.setVelocity(left_cmd, right_cmd)

    def update_odometry(self, dt):
        left_ticks, right_ticks = self.pibot.getEncoders()
        dl = left_ticks - self.prev_left_ticks
        dr = right_ticks - self.prev_right_ticks
        self.prev_left_ticks, self.prev_right_ticks = left_ticks, right_ticks

        left_dist = 2*math.pi*self.WHEEL_RADIUS * dl / self.TICKS_PER_REV
        right_dist = 2*math.pi*self.WHEEL_RADIUS * dr / self.TICKS_PER_REV

        linear = (left_dist + right_dist) / 2.0
        delta_th = (right_dist - left_dist) / self.wheel_base

        self.current_pose[2] = (self.current_pose[2] + delta_th + np.pi) % (2*np.pi) - np.pi
        th = self.current_pose[2]

        self.current_pose[0] += linear * math.cos(th)
        self.current_pose[1] += linear * math.sin(th)

        self.trajectory.append(self.current_pose.copy())
        
        # Debug print
        print(f"📍 Current pose: x={self.current_pose[0]:.2f}m, y={self.current_pose[1]:.2f}m, θ={math.degrees(self.current_pose[2]):.1f}°")

    def move_to_waypoint(self, waypoint, dt=0.1, desired_speed=0.2):
        start = time.time()
        self.integral_yaw = 0.0
        self.prev_yaw_error = 0.0
        
        print(f"🎯 Navigating to waypoint: {waypoint}")

        while time.time() - start < self.time_limit:
            xC, yC, thC = self.current_pose
            dist_to_wp = np.linalg.norm([waypoint[0] - xC, waypoint[1] - yC])
            
            if dist_to_wp < self.position_threshold:
                print(f"✅ Reached {waypoint}")
                break

            dx = waypoint[0] - xC
            dy = waypoint[1] - yC
            desired_heading = math.atan2(dy, dx)
            yaw_error = (desired_heading - thC + np.pi) % (2*np.pi) - np.pi

            self.integral_yaw += yaw_error * dt
            deriv = (yaw_error - self.prev_yaw_error) / dt
            rot_cmd = self.Kp*yaw_error + self.Ki*self.integral_yaw + self.Kd*deriv
            self.prev_yaw_error = yaw_error

            # More aggressive scaling for turns
            dist_scale = min(1.0, dist_to_wp/0.5)
            head_scale = max(0.2, 1 - abs(yaw_error)/(math.pi/2))  # More aggressive
            forward_cmd = desired_speed * dist_scale * head_scale

            self.set_wheel_speeds_forward(forward_cmd, rot_cmd)
            self.update_odometry(dt)
            time.sleep(dt)


    def run_controller(self, dt=0.1, desired_speed=0.2):
        print("Starting pure pursuit...")
        self.move_to_waypoint(self.target_point, dt, desired_speed)
        self.pibot.setVelocity(0, 0)
        print("Done.")

# --- Fire detection and position estimation ---
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

def estimate_position(cx, cy, frame_width):
    """Estimate fire position label based on image x-coordinate"""
    if cx < frame_width / 3:
        return "left"
    elif cx > 2 * frame_width / 3:
        return "right"
    else:
        return "center"

# --- Navigation ---
def navigate_to_beacon(pibot, controller, position_label):
    """Navigate to a known beacon position using odometry"""
    if position_label not in KNOWN_BEACON_POSITIONS:
        print(f"❌ Unknown position label: {position_label}")
        return
    
    target_point = KNOWN_BEACON_POSITIONS[position_label]
    print(f"🧭 Navigating to beacon at {position_label} (x={target_point[0]:.2f}m, y={target_point[1]:.2f}m)...")
    
    # Use the existing controller
    controller.move_to_waypoint(target_point)

def main(pibot: PiBot):
    print("🔥 Firefighter Robot Activated")
    
    # Initialize controller with a dummy target (will be overwritten in move_to_waypoint)
    controller = SimplePiBotController(pibot, target_point=(0,0))
    
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

    # Sort fires by area (biggest first)
    fire_targets.sort(key=lambda x: x['area'], reverse=True)

    print("\n🧠 Fire detection complete. Visiting in order of intensity:")
    for i, fire in enumerate(fire_targets):
        print(f"  🔥 Fire {i+1} at {fire['position_label']} | Intensity: {int(fire['area'])}")

    # --- VISIT EACH FIRE ---
    for i, fire in enumerate(fire_targets):
        pos = fire['position_label']
        print(f"\n🚒 Addressing fire {i+1} at {pos} | Intensity: {int(fire['area'])}")
        
        if pos not in KNOWN_BEACON_POSITIONS:
            print(f"❌ Unknown position label: {pos}")
            continue
            
        navigate_to_beacon(pibot, controller, pos)
        
        # Small pause between waypoints
        time.sleep(0.5)

    print("\n✅ All fires addressed.")
    pibot.setVelocity(0, 0)
    cv.destroyAllWindows()



if __name__ == "__main__":
    pibot = PiBot(ip="172.19.232.191", port=8080, localiser_ip=None)
    main(pibot)
