#!/usr/bin/env python3
"""Top-level rover bringup: camera + ESP32 motor link, in one command.

Runs ON THE PI. Launch by full path -- no colcon build needed:

    source /opt/ros/humble/setup.bash
    source ~/rover_ws/install/setup.bash
    source ~/uros_ws/install/setup.bash          # provides micro_ros_agent
    ros2 launch ~/rover_ws/src/omnivla_nav/launch/rover_master_bringup.launch.py

There is no "master" to start: ROS 2 replaced roscore with DDS discovery, so
this file IS the top level -- nothing needs to run before it.

Keyboard driving is deliberately NOT started here. A launch-managed process
does not get the terminal's raw stdin, so a teleop node started this way would
come up and then ignore every keypress. Run it in its own terminal instead:

    ~/ros_keyboard_control/run_teleop.sh ros2

Arguments
    use_camera:=false     REQUIRED when recording with ffmpeg -- the camera node
                          holds /dev/video0 exclusively and ffmpeg will fail with
                          "Device or resource busy".
    use_agent:=false      Skip the ESP32 link (camera-only bringup).
    image_size:=[1280,720]
    pixel_format:=MJPG    Never YUYV above 640x480 -- it drops to 5 fps.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    video_device = LaunchConfiguration("video_device").perform(context)
    pixel_format = LaunchConfiguration("pixel_format").perform(context)
    image_size = LaunchConfiguration("image_size").perform(context)
    publish_rate = LaunchConfiguration("publish_rate").perform(context)
    serial_port = LaunchConfiguration("serial_port").perform(context)
    baud = LaunchConfiguration("baud").perform(context)
    use_camera = LaunchConfiguration("use_camera").perform(context).lower() == "true"
    use_agent = LaunchConfiguration("use_agent").perform(context).lower() == "true"

    try:
        dims = eval(image_size)
        width, height = int(dims[0]), int(dims[1])
    except Exception:
        width, height = 1280, 720

    import os
    os.environ["ROS_LOCALHOST_ONLY"] = "0"
    os.environ.setdefault("ROS_DOMAIN_ID", "0")

    actions = []

    if use_camera:
        actions += [
            LogInfo(msg=f"[bringup] camera: {video_device} {pixel_format} "
                        f"{width}x{height} @ {publish_rate} Hz -> /image_raw"),
            LogInfo(msg="[bringup] NOTE: this node holds /dev/video0 exclusively. "
                        "To record with ffmpeg, relaunch with use_camera:=false"),
            Node(
                package="v4l2_camera",
                executable="v4l2_camera_node",
                name="v4l2_camera",
                output="screen",
                parameters=[{
                    "video_device": video_device,
                    "pixel_format": pixel_format,
                    "image_size": [width, height],
                    "publish_rate": float(publish_rate),
                }],
            ),
        ]
    else:
        actions.append(LogInfo(
            msg="[bringup] camera disabled -- /dev/video0 free for ffmpeg"))

    if use_agent:
        actions += [
            LogInfo(msg=f"[bringup] micro-ROS agent on {serial_port} @ {baud} "
                        f"(ESP32: subscribes /cmd_vel, publishes /rover/rpm)"),
            Node(
                package="micro_ros_agent",
                executable="micro_ros_agent",
                name="micro_ros_agent",
                output="screen",
                arguments=["serial", "--dev", serial_port, "-b", baud, "-v4"],
            ),
        ]
    else:
        actions.append(LogInfo(msg="[bringup] ESP32 agent disabled"))

    actions.append(LogInfo(
        msg="[bringup] drive it from ANOTHER terminal: "
            "~/ros_keyboard_control/run_teleop.sh ros2   "
            "(tap 'z' twice for ~0.2 m/s before recording SLAM footage)"))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("video_device", default_value="/dev/video0"),
        DeclareLaunchArgument("pixel_format", default_value="MJPG",
                              description="MJPG or YUYV; YUYV drops to 5fps >640x480"),
        DeclareLaunchArgument("image_size", default_value="[1280,720]"),
        DeclareLaunchArgument("publish_rate", default_value="30.0"),
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("baud", default_value="115200"),
        DeclareLaunchArgument("use_camera", default_value="true",
                              description="false frees /dev/video0 for ffmpeg"),
        DeclareLaunchArgument("use_agent", default_value="true",
                              description="false skips the ESP32 motor link"),
        OpaqueFunction(function=launch_setup),
    ])
