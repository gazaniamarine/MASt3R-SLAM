#!/usr/bin/env python3
# ==============================================================================
#  ROS2 Keyboard Teleoperation Node
# ==============================================================================
import sys
import os
import select
import termios
import tty
import time
from threading import Thread, Lock

# Standard ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
except ImportError:
    print("\033[1;31mERROR: rclpy or geometry_msgs not found!\033[0m")
    print("Please make sure ROS2 is sourced (e.g., source /opt/ros/jazzy/setup.bash)")
    sys.exit(1)

# Keyboard map configuration
MOVE_BINDINGS = {
    'w': (1.0, 0.0),   # Forward
    's': (-1.0, 0.0),  # Backward
    'a': (0.0, 1.0),   # Turn Left
    'd': (0.0, -1.0),  # Turn Right
}

SPEED_BINDINGS = {
    'q': (1.1, 1.0),   # Increase linear speed by 10%
    'z': (0.9, 1.0),   # Decrease linear speed by 10%
    'e': (1.0, 1.1),   # Increase angular speed by 10%
    'c': (1.0, 0.9),   # Decrease angular speed by 10%
}

# ANSI Escape Codes for UI styling
CLEAR_SCREEN = "\033[2J\033[H"
COLOR_HEADER = "\033[1;36m"
COLOR_LABEL = "\033[1;33m"
COLOR_VALUE = "\033[1;32m"
COLOR_WARN = "\033[1;31m"
COLOR_INFO = "\033[1;34m"
COLOR_RESET = "\033[0m"
BOLD = "\033[1m"

HELP_TEXT = """
┌──────────────────────────────────────────────────────────┐
│              ROS2 KEYBOARD TELEOP CONTROLLER             │
├──────────────────────────────────────────────────────────┤
│  Movement:                      Speed Adjustment:        │
│      w : Forward                  q / z : Lin speed +/-  │
│      s : Backward                 e / c : Ang speed +/-  │
│      a : Turn Left                                       │
│      d : Turn Right             Emergency Stop:          │
│                                   Space / x : Stop       │
│  Quit: Ctrl + C                                          │
└──────────────────────────────────────────────────────────┘
"""

class KeyboardTeleopNode(Node):
    def __init__(self):
        super().__init__('keyboard_teleop_node')
        
        # ROS2 Publisher
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Safe defaults
        self.max_lin_speed = 0.4   # m/s
        self.max_ang_speed = 0.6   # rad/s
        
        self.target_lin = 0.0
        self.target_ang = 0.0
        self.last_key = 'None'
        self.lock = Lock()
        self.running = True
        
        # Start command publisher loop at 10Hz
        self.pub_thread = Thread(target=self.publisher_loop, daemon=True)
        self.pub_thread.start()

    def update_velocity(self, key):
        with self.lock:
            self.last_key = key
            
            # Movement keys
            if key in MOVE_BINDINGS:
                lin_dir, ang_dir = MOVE_BINDINGS[key]
                # Combine linear and angular inputs nicely
                if lin_dir != 0:
                    self.target_lin = lin_dir * self.max_lin_speed
                if ang_dir != 0:
                    self.target_ang = ang_dir * self.max_ang_speed
            
            # Speed adjustment keys
            elif key in SPEED_BINDINGS:
                lin_factor, ang_factor = SPEED_BINDINGS[key]
                self.max_lin_speed = max(0.05, min(1.5, self.max_lin_speed * lin_factor))
                self.max_ang_speed = max(0.1, min(2.5, self.max_ang_speed * ang_factor))
            
            # Stop keys
            elif key in (' ', 'x'):
                self.target_lin = 0.0
                self.target_ang = 0.0
                
    def stop_robot(self):
        with self.lock:
            self.target_lin = 0.0
            self.target_ang = 0.0

    def publisher_loop(self):
        rate = 0.1  # 10 Hz
        while self.running and rclpy.ok():
            with self.lock:
                twist = Twist()
                twist.linear.x = float(self.target_lin)
                twist.angular.z = float(self.target_ang)
                
            try:
                self.pub_cmd.publish(twist)
            except Exception as e:
                pass
            time.sleep(rate)

    def draw_tui(self):
        # Create a beautiful terminal visual
        with self.lock:
            v_lin = self.target_lin
            v_ang = self.target_ang
            limit_lin = self.max_lin_speed
            limit_ang = self.max_ang_speed
            key_pressed = self.last_key
            
        sys.stdout.write(CLEAR_SCREEN)
        sys.stdout.write(COLOR_HEADER + HELP_TEXT + COLOR_RESET)
        
        # Dynamic direction helper
        direction = "STOPPED"
        if v_lin > 0:
            direction = "▲ FORWARD"
        elif v_lin < 0:
            direction = "▼ BACKWARD"
        elif v_ang > 0:
            direction = "◀ LEFT TURN"
        elif v_ang < 0:
            direction = "▶ RIGHT TURN"
            
        sys.stdout.write(f"\n{COLOR_LABEL}--- ACTIVE STATUS ---{COLOR_RESET}\n")
        sys.stdout.write(f"  Last key pressed   : {COLOR_VALUE}{key_pressed}{COLOR_RESET}\n")
        sys.stdout.write(f"  Motion Direction   : {COLOR_VALUE}{direction}{COLOR_RESET}\n\n")
        
        sys.stdout.write(f"{COLOR_LABEL}--- SPEED CONTROL ---{COLOR_RESET}\n")
        sys.stdout.write(f"  Target Linear Vel  : {COLOR_VALUE}{v_lin:+.2f} m/s{COLOR_RESET}  [Max limit: {limit_lin:.2f} m/s]\n")
        sys.stdout.write(f"  Target Angular Vel : {COLOR_VALUE}{v_ang:+.2f} rad/s{COLOR_RESET} [Max limit: {limit_ang:.2f} rad/s]\n")
        
        # Visual progress bar for speeds
        lin_bar = "#" * int(abs(v_lin) / 1.5 * 20) + "-" * (20 - int(abs(v_lin) / 1.5 * 20))
        ang_bar = "#" * int(abs(v_ang) / 2.5 * 20) + "-" * (20 - int(abs(v_ang) / 2.5 * 20))
        
        sys.stdout.write(f"\n{COLOR_LABEL}--- VISUAL LEVEL METERS ---{COLOR_RESET}\n")
        sys.stdout.write(f"  Linear  : [{COLOR_VALUE}{lin_bar}{COLOR_RESET}] {abs(v_lin):.2f} m/s\n")
        sys.stdout.write(f"  Angular : [{COLOR_VALUE}{ang_bar}{COLOR_RESET}] {abs(v_ang):.2f} rad/s\n")
        
        sys.stdout.write(f"\n{COLOR_WARN}Press Ctrl+C to terminate keyboard control and safely stop the robot.{COLOR_RESET}\n")
        sys.stdout.flush()

def getKey(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main():
    settings = termios.tcgetattr(sys.stdin)
    
    rclpy.init()
    node = KeyboardTeleopNode()
    
    # Run the TUI draw loop in a separate thread
    def tui_loop():
        while node.running:
            node.draw_tui()
            time.sleep(0.1)
            
    tui_thread = Thread(target=tui_loop, daemon=True)
    tui_thread.start()
    
    try:
        while True:
            key = getKey(settings)
            if key == '\x03':  # Ctrl+C
                break
            if key != '':
                node.update_velocity(key)
            else:
                # Optional: Uncomment if you want automatic speed decay/stopping on no keypresses
                # For standard keyboard teleop, keeping speed set until space/x is preferred.
                pass
                
    except Exception as e:
        print(e)
    finally:
        node.running = False
        node.stop_robot()
        
        # Publish final zero command
        try:
            twist = Twist()
            node.pub_cmd.publish(twist)
        except Exception:
            pass
            
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        print("\n\033[1;32mSafely stopped the robot. Exiting keyboard teleop.\033[0m")
        rclpy.shutdown()

if __name__ == '__main__':
    main()
