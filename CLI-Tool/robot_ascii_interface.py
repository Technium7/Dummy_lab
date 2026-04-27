#!/usr/bin/env python3
"""
Robot ASCII Protocol Interface
Provides a simple Python interface to send ASCII commands to the DummyRobot over COM3.
"""

import serial
import time
import sys


class RobotAsciiInterface:
    def __init__(self, port='COM3', baudrate=115200, timeout=1.0):
        """
        Initialize serial connection to robot.
        
        Args:
            port: Serial port name (default: COM3)
            baudrate: Baud rate (default: 115200)
            timeout: Read timeout in seconds (default: 1.0)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        
    def connect(self):
        """Open serial connection."""
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.timeout
            )
            print(f"Connected to {self.port} at {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"Error connecting to {self.port}: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Disconnected")

    def _clear_stale_input(self):
        """Clear unread serial bytes to avoid mixing old responses with new commands."""
        if not self.ser or not self.ser.is_open:
            return
        try:
            if self.ser.in_waiting > 0:
                self.ser.reset_input_buffer()
        except serial.SerialException:
            pass
    
    def send_command(self, cmd, wait_ms=200):
        """
        Send a command and get response.
        
        Args:
            cmd: Command string (e.g., "!START", "#GETJPOS", "&10,0,90,0,0,0,20")
            wait_ms: Wait time before reading response (milliseconds)
            
        Returns:
            Response string or None if timeout
        """
        if not self.ser or not self.ser.is_open:
            print("Not connected")
            return None

        self._clear_stale_input()
        
        # Send command with newline
        self.ser.write((cmd + '\n').encode())
        self.ser.flush()
        
        # Wait for response
        time.sleep(wait_ms / 1000.0)
        
        try:
            responses = []
            deadline = time.time() + max(self.timeout, 0.3)

            while time.time() < deadline:
                raw = self.ser.readline()
                if not raw:
                    if responses:
                        break
                    continue

                response = raw.decode('utf-8', errors='ignore').strip().replace('\x00', '')
                if response:
                    responses.append(response)

                # Handle commands that produce two quick lines, e.g. '&' => '15' then 'ok'
                time.sleep(0.03)
                if self.ser.in_waiting == 0 and responses:
                    break

            if not responses:
                return None

            return responses[-1]
        except (serial.SerialException, UnicodeDecodeError) as e:
            print(f"Error reading response: {e}")
            return None
    
    # Control Commands
    def start(self):
        """Enable robot."""
        resp = self.send_command('!START')
        return resp
    
    def stop(self):
        """Emergency stop."""
        resp = self.send_command('!STOP')
        return resp
    
    def home(self):
        """Home all joints to (0, 0, 90, 0, 0, 0)."""
        resp = self.send_command('!HOME', wait_ms=2000)
        return resp
    
    def reset(self):
        """Reset to rest position."""
        resp = self.send_command('!RESET', wait_ms=2000)
        return resp
    
    def disable(self):
        """Disable motors."""
        resp = self.send_command('!DISABLE')
        return resp
    
    # Query Commands
    def get_joint_pos(self):
        """Get current joint positions (J1-J6 in degrees)."""
        # Use a shorter wait for high-rate position polling.
        resp = self.send_command('#GETJPOS', wait_ms=100)
        if resp and resp.startswith('ok'):
            try:
                values = resp.split()[1:]  # Skip 'ok'
                if len(values) < 6:
                    return None
                joints = [float(v) for v in values[:6]]
                return joints
            except (ValueError, IndexError):
                return None
        return None
    
    def get_linear_pos(self):
        """Get current linear position (X, Y, Z, A, B, C)."""
        resp = self.send_command('#GETLPOS')
        if resp and resp.startswith('ok'):
            try:
                values = resp.split()[1:]  # Skip 'ok'
                if len(values) < 6:
                    return None
                pose = [float(v) for v in values[:6]]
                return pose
            except (ValueError, IndexError):
                return None
        return None
    
    def set_command_mode(self, mode):
        """
        Set command mode.
        
        Args:
            mode: 1=Sequential, 2=Interruptable, 3=Trajectory, 4=Tuning
        """
        resp = self.send_command(f'#CMDMODE {mode}')
        return resp
    
    # Motion Commands
    def send_trajectory_point_async(self, j1, j2, j3, j4, j5, j6, speed):
        """
        Send a single trajectory waypoint WITHOUT waiting for response.
        Designed for continuous streaming to avoid braking between waypoints.

        Returns:
            True if sent successfully, False on error
        """
        if not self.ser or not self.ser.is_open:
            return False

        try:
            cmd = f'&{j1:.3f},{j2:.3f},{j3:.3f},{j4:.3f},{j5:.3f},{j6:.3f},{speed:.1f}'
            self.ser.write((cmd + '\n').encode())
            self.ser.flush()
            return True
        except serial.SerialException:
            return False

    def send_trajectory_point(self, j1, j2, j3, j4, j5, j6, speed, timeout_sec=30):
        """
        Send a single trajectory waypoint and block until robot confirms 'ok'.
        Designed for use inside run_trajectory - does NOT clear the input buffer
        between waypoints so streaming stays tight.

        Returns:
            True if 'ok' received, False on timeout/error
        """
        if not self.ser or not self.ser.is_open:
            return False

        cmd = f'&{j1:.3f},{j2:.3f},{j3:.3f},{j4:.3f},{j5:.3f},{j6:.3f},{speed:.1f}'
        self.ser.write((cmd + '\n').encode())
        self.ser.flush()

        start = time.time()
        while time.time() - start < timeout_sec:
            try:
                raw = self.ser.readline()
                if raw:
                    line = raw.decode('utf-8', errors='ignore').strip()
                    if 'ok' in line:
                        return True
            except serial.SerialException:
                return False
        return False

    def run_trajectory(self, start_joints, end_joints,
                       n_points=30, min_speed=5, max_speed=40,
                       send_interval_ms=50, timeout_sec=60):
        """
        Execute a smooth point-to-point trajectory using CMDMODE 3.
        Velocity follows a bell (sine) curve: ramps up, cruises, ramps down.

        Args:
            start_joints : list of 6 floats – starting joint angles (degrees)
            end_joints   : list of 6 floats – target  joint angles (degrees)
            n_points     : number of waypoints (more = smoother, but higher latency)
            min_speed    : speed at start/end of trajectory (degrees/s)
            max_speed    : peak speed at midpoint (degrees/s)
            timeout_sec  : per-waypoint timeout

        Returns:
            True if all waypoints were acknowledged, False otherwise
        """
        import math

        print(f"Running trajectory: {n_points} waypoints, speed {min_speed}→{max_speed}→{min_speed}")

        for i in range(1, n_points + 1):
            t = i / n_points  # 0 → 1

            # Bell-shaped speed profile (sine arch: 0 → 1 → 0)
            speed = min_speed + (max_speed - min_speed) * math.sin(math.pi * t)
            speed = max(min_speed, speed)

            # Linear position interpolation
            joints = [
                start_joints[j] + (end_joints[j] - start_joints[j]) * t
                for j in range(6)
            ]

            pct = int(t * 100)
            print(f"  [{pct:3d}%] J1={joints[0]:6.2f} spd={speed:.1f}", end='  ')

            # Send WITHOUT waiting for response (async) to avoid braking
            ok = self.send_trajectory_point_async(
                joints[0], joints[1], joints[2],
                joints[3], joints[4], joints[5],
                speed
            )

            if ok:
                print("sent")
            else:
                print("SEND FAILED")
                return False

            # Delay before next waypoint to maintain streaming
            time.sleep(send_interval_ms / 1000.0)

        # All waypoints sent; wait for final motion to complete
        print("  Waiting for trajectory to complete...")
        time.sleep(2.0)
        return True

    def move_joint(self, j1, j2, j3, j4, j5, j6, speed=None, wait_ms=500):
        """
        Move to joint position.
        
        Args:
            j1-j6: Joint angles in degrees
            speed: Optional speed (0-100, degree/s)
            wait_ms: Wait time before reading response
        """
        if speed is not None:
            cmd = f'&{j1},{j2},{j3},{j4},{j5},{j6},{speed}'
        else:
            cmd = f'&{j1},{j2},{j3},{j4},{j5},{j6}'
        
        resp = self.send_command(cmd, wait_ms=wait_ms)
        return resp
    
    def move_linear(self, x, y, z, a, b, c, speed=None, wait_ms=500):
        """
        Move to linear position (if supported by robot).
        
        Args:
            x, y, z: Position in mm
            a, b, c: Orientation in degrees
            speed: Optional speed
            wait_ms: Wait time before reading response
        """
        if speed is not None:
            cmd = f'@{x},{y},{z},{a},{b},{c},{speed}'
        else:
            cmd = f'@{x},{y},{z},{a},{b},{c}'
        
        resp = self.send_command(cmd, wait_ms=wait_ms)
        return resp
    
    def wait_for_motion_complete(self, timeout_sec=30, poll_interval_ms=100, tolerance=0.5):
        """
        Wait for robot motion to complete by polling position until stable.
        
        Args:
            timeout_sec: Maximum time to wait (seconds)
            poll_interval_ms: Poll interval (milliseconds)
            tolerance: Position change threshold to detect motion (degrees)
            
        Returns:
            True if motion completed, False if timeout
        """
        import time
        
        start_time = time.time()
        last_pos = None
        stable_count = 0
        stable_threshold = 3  # Require 3 consecutive stable readings
        
        while time.time() - start_time < timeout_sec:
            pos = self.get_joint_pos()
            
            if pos is None or len(pos) < 6:
                time.sleep(poll_interval_ms / 1000.0)
                continue
            
            # Check if position changed significantly
            if last_pos is None or len(last_pos) < 6:
                last_pos = pos
                stable_count = 0
            else:
                # Calculate max difference between current and last position
                try:
                    max_diff = max(abs(pos[i] - last_pos[i]) for i in range(6))
                    
                    if max_diff < tolerance:
                        stable_count += 1
                        if stable_count >= stable_threshold:
                            return True  # Motion complete
                    else:
                        stable_count = 0  # Reset counter if motion detected
                        last_pos = pos
                except (TypeError, ValueError):
                    # If comparison fails, skip this poll
                    time.sleep(poll_interval_ms / 1000.0)
                    continue
            
            time.sleep(poll_interval_ms / 1000.0)
        
        return False  # Timeout
    
    def interactive_shell(self):
        """Run interactive command shell."""
        print("Robot ASCII Interface - Interactive Shell")
        print("Type 'help' for available commands, 'exit' to quit")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("robot> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'exit':
                    break
                
                if user_input.lower() == 'help':
                    print("""
Available Commands:
  Control:
    !START              - Enable robot
    !STOP               - Emergency stop
    !HOME               - Home position
    !RESET              - Rest position
    !DISABLE            - Disable motors
    
  Query:
    #GETJPOS            - Get joint positions
    #GETLPOS            - Get linear position
    #CMDMODE <mode>     - Set mode (1-4)
    
  Motion (joint):
    &j1,j2,j3,j4,j5,j6[,speed]
    Example: &0,0,90,0,0,0,20
    
  Motion (linear):
    @x,y,z,a,b,c[,speed]
    Example: @100,200,300,0,0,0
""")
                    continue
                
                # Send raw command
                resp = self.send_command(user_input)
                if resp:
                    print(f"Response: {resp}")
                else:
                    print("No response (timeout)")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    # Create interface
    robot = RobotAsciiInterface()
    
    # Connect
    if not robot.connect():
        sys.exit(1)
    
    try:
        # Run interactive shell
        robot.interactive_shell()
    finally:
        robot.disconnect()


if __name__ == '__main__':
    main()
