from ultralytics import YOLO
import cv2
import torch
import keyboard
import pyrealsense2 as rs
import numpy as np

import socket

safe_distance = False

# TODO
# - Realsense Camera setup instead of laptop webcam      DONE
# - Conversion of pixel to depth with RealSense          DONE - but needs testing/confirmation
# - Create UDP connection to robot (then have robot stop motion)        WIP
# - Padding of limbs annotation with mask processing        BONUS if time

# Setup RealSense Camera
def realsenseInit():
    pipe = rs.pipeline()
    cfg = rs.config() #for configuring the camera preferences

    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipe.start(cfg)

    return pipe




# Apply Model to predict Pose
def applyModelToCapture(image, device):
    """
    Retrieve keypoint (x,y) tuple list for the body pose

    Args:
        image:
            numpy array for the raw RGB image captured by Realsense Camera
        device:
            device being used for processing, "cuda" if available, otherwise "cpu"

    Returns:
        List of keypoint (x,y) tuples detected by YOLO model.
            For the default pose model, keypoint indices for human body pose estimation are:
               0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
               5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
               9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
               13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
    """
    pose_result = model.predict(image, show=True, device=device)  # Process frame directly
    pose_keypoints = pose_result[0].keypoints.xy #keypoints of pose (pixels)

    return pose_keypoints


def findPoseLines(pose_keypoints_pixels, original_image):
    """
    Draw detected skeleton of pose from realsense capture onto black image and determines all the pixels for the lines.

    Args:
        pose_keypoints_pixels: Keypoints detected by YOLO model
            For the default pose model, keypoint indices for human body pose estimation are:
               0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
               5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
               9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
               13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
        original_image: Original color image from the camera

    Returns:
        Black image with skeleton lines, and list of points making up the lines detected by YOLO model.
    """

    def is_valid(point):
        return 0 < point[0] < width and 0 < point[1] < height

    height, width = original_image.shape[:2]
    black_image = np.zeros((height, width, 3), np.uint8) # image to store the cv2 line coordinates and return them
    keypoints = pose_keypoints_pixels

    # Tensor has shape [1, 17, 2]: # People detected, # Keypoints per person, #XY coordinates per point
    if torch.is_tensor(pose_keypoints_pixels):
        keypoints = pose_keypoints_pixels.cpu().numpy()

    try:
        # left arm
        left_wrist = keypoints[0][9].astype(int)
        left_elbow = keypoints[0][7].astype(int)
        left_shoulder = keypoints[0][5].astype(int)

        # right arm
        right_wrist = keypoints[0][10].astype(int)
        right_elbow = keypoints[0][8].astype(int)
        right_shoulder = keypoints[0][6].astype(int)

        # left leg
        left_ankle = keypoints[0][15].astype(int)
        left_knee = keypoints[0][13].astype(int)
        left_hip = keypoints[0][11].astype(int)

        # right leg
        right_ankle = keypoints[0][16].astype(int)
        right_knee = keypoints[0][14].astype(int)
        right_hip = keypoints[0][12].astype(int)


        # Left forearm
        if is_valid(left_wrist) and is_valid(left_elbow):
            cv2.line(black_image,
                     (int(left_elbow[0]), int(left_elbow[1])),
                     (int(left_wrist[0]), int(left_wrist[1])),
                     (255, 255, 255), 2)

        # Left upper arm
        if is_valid(left_shoulder) and is_valid(left_elbow):
            cv2.line(black_image,
                     (int(left_shoulder[0]), int(left_shoulder[1])),
                     (int(left_elbow[0]), int(left_elbow[1])),
                     (255, 255, 255), 2)

        # Right forearm
        if is_valid(right_elbow) and is_valid(right_wrist):
            cv2.line(black_image,
                     (int(right_elbow[0]), int(right_elbow[1])),
                     (int(right_wrist[0]), int(right_wrist[1])),
                     (255, 255, 255), 2)

        # Right upper arm
        if is_valid(right_elbow) and is_valid(right_shoulder):
            cv2.line(black_image,
                     (int(right_shoulder[0]), int(right_shoulder[1])),
                     (int(right_elbow[0]), int(right_elbow[1])),
                     (255, 255, 255), 2)

        # Left shin
        if is_valid(left_ankle) and is_valid(left_knee):
            cv2.line(black_image,
                     (int(left_ankle[0]), int(left_ankle[1])),
                     (int(left_knee[0]), int(left_knee[1])),
                     (255, 255, 255),2)

        # Left quad
        if is_valid(left_knee) and is_valid(left_hip):
            cv2.line(black_image,
                     (int(left_knee[0]), int(left_knee[1])),
                     (int(left_hip[0]), int(left_hip[1])),
                     (255, 255, 255), 2)

        # Right shin
        if is_valid(right_ankle) and is_valid(right_knee):
            cv2.line(black_image,
                     (int(right_ankle[0]), int(right_ankle[1])),
                     (int(right_knee[0]), int(right_knee[1])),
                     (255, 255, 255), 2)

        # Right quad
        if is_valid(right_knee) and is_valid(right_hip):
            cv2.line(black_image,
                     (int(right_knee[0]), int(right_knee[1])),
                     (int(right_hip[0]), int(right_hip[1])),
                     (255, 255, 255), 2)

        # Shoulder profile
        if is_valid(right_shoulder) and is_valid(left_shoulder):
            cv2.line(black_image,
                     (int(right_shoulder[0]), int(right_shoulder[1])),
                     (int(left_shoulder[0]), int(left_shoulder[1])),
                     (255, 255, 255), 2)

        # Shoulder profile
        if is_valid(right_hip) and is_valid(left_hip):
            cv2.line(black_image,
                     (int(right_hip[0]), int(right_hip[1])),
                     (int(left_hip[0]), int(left_hip[1])),
                     (255, 255, 255), 2)

        # Torso sides
        if is_valid(right_hip) and is_valid(left_hip) and is_valid(left_shoulder) and is_valid(right_shoulder):
            cv2.line(black_image,
                     (int(right_hip[0]), int(right_hip[1])),
                     (int(right_shoulder[0]), int(right_shoulder[1])),
                     (255, 255, 255), 2)
            cv2.line(black_image,
                     (int(left_hip[0]), int(left_hip[1])),
                     (int(left_shoulder[0]), int(left_shoulder[1])),
                     (255, 255, 255), 2)

    except (IndexError, TypeError, ValueError) as e:
        print(f"Error processing keypoints: {e}")
        return black_image, None

    # coordinates where limbs are detected
    y_coords, x_coords, _ = np.where(black_image == 255)
    line_pixels = list(set(zip(y_coords, x_coords)))

    return black_image, line_pixels


if __name__ == "__main__":
    # Load the YOLO model
    model = YOLO("yolo11n-pose.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    UDP_IP = "172.31.1.147"  # IP of the robot
    UDP_PORT = 30001 # Port of the robot???
    sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)  # Create UDP socket
    # sock.bind((UDP_IP, UDP_PORT))
    print("*** UDP Established ***")


    # Starting RealSense Camera
    pipe = realsenseInit()
    threshold = 400 # millimeter distance limit on depth before stopping robot. Modify if needed

    while True:
        # Getting RealSense frames
        frame_rs = pipe.wait_for_frames()
        color_rs = frame_rs.get_color_frame()
        depth_rs = frame_rs.get_depth_frame()
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_rs, alpha = 0.5), cv2.COLORMAP_JET)

        # Converting RealSense frame to arrays
        color_array = np.asanyarray(color_rs.get_data()) # RGB image
        depth_array = np.asanyarray(depth_rs.get_data()) # Depth imagez

        # cv2.imshow("Color Frame", color_array)
            # cv2.imshow("Depth Frame", depth_array)

        if color_array is not None:
            pose_keypoints_pixels = applyModelToCapture(color_array, device)  # Run model on live frame

            new_image, line_pixels = findPoseLines(pose_keypoints_pixels, color_array)
            cv2.imshow("Pose Lines", new_image)

            if line_pixels is not None:
                valid_depths = [depth_array[y][x] for y, x in line_pixels if depth_array[y][x] > 0]
                if len(valid_depths) < 10:  # Adjust this number based on your needs
                    safe_distance = False
                else:
                    safe_distance = all(depth > threshold for depth in valid_depths)
                print(f"Safe: {safe_distance}, Min depth: {min(valid_depths, default=0)} mm")
                try:
                    sock.sendto(str(safe_distance).encode('utf-8'), (UDP_IP, UDP_PORT))
                    print("UDP Sent SAFE/UNSAFE")
                except:
                    print("Error in UDP communication (POSE DETECTED)")
            else:
                print("POSE NOT DETECTED")
                try:
                    sock.sendto("POSE NOT DETECTED".encode('utf-8'), (UDP_IP, UDP_PORT))
                    print("UDP Sent POSE NOT DETECTED")
                except:
                    print("Error in UDP communication (POSE NOT DETECTED)")

        if cv2.waitKey(1) & 0xFF == ord('q')  or keyboard.is_pressed('e'):
            break

    pipe.stop()
    cv2.destroyAllWindows()
