import cv2
import mediapipe as mp
import numpy as np
import json
import time
import socket
import sys

# color in BGR
text_color = (255, 255, 255)
arrow_color = (0, 0, 255)
arrow_thickness = 6 #px
arrow_length = 200
cmd_text_position = [50,50]
last_cmd_expire_time = {}
ports_text_position = [100,100]
all_cmd = set()

current_states = {
    'action': None,
    'event_type': 'tracking',
    'displaying_action': None,
}

def read_configs(file_name):
    f = open(file_name)
    # returns JSON object as a dictionary
    actions = json.load(f)

    f.close()
    return actions

# Calculate angles
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_arrow_endpoints(direction, angle, anchors, length = arrow_length):
    # TODO: calculate realtime end point after arm orientation is changed
    if len(anchors) == 1:
        start_point = anchors[0]
    elif len(anchors) > 0:
        start_point = np.average(np.array(anchors), 0)
    start_point = tuple(np.multiply(start_point, [1280, 720]).astype(int))
    if direction == "right":
        end_point = (start_point[0]-length, start_point[1])
    return start_point, end_point

def calculate_circle(image,circle_direction, angle, anchors, circle_center, length = arrow_length):
    center_point = tuple(np.multiply(circle_center, [1280, 720]).astype(int))
    radians = np.arctan2(anchors[0][1] - anchors[1][1], anchors[0][0] - anchors[1][0])
    arm_angle =radians * 180.0 / np.pi
    radius = 150
    axes = (radius, radius)
    startAngle = arm_angle
    delta =60
    if circle_direction == "down" and arm_angle < 0:
        delta *= -1
    thickness = 16
    WHITE = (255, 255, 255)
    cv2.ellipse(image, center_point, axes, startAngle, 0, delta, WHITE, thickness)



def get_anchors(action_def, all_coorddinates):
    anchors_str = action_def["illustration"]["anchors"]
    anchors_point = [all_coorddinates[point_str] for point_str in anchors_str]
    return anchors_point

def visualize_angles (landmarks, mp_pose, image):
    # Get coordinates
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    all_coorddinates = {
        "left_hip": left_hip,
        "left_shoulder": left_shoulder,
        "left_elbow": left_elbow,
        "left_wrist": left_wrist,
        "left_knee": left_knee,
        "left_ankle": left_ankle,
        "right_hip": right_hip,
        "right_shoulder": right_shoulder,
        "right_elbow": right_elbow,
        "right_wrist": right_wrist,
        "right_knee": right_knee,
        "right_ankle": right_ankle,
    }

    # Calculate angle
    left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
    right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)

    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    left_leg_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_leg_angle = calculate_angle(right_shoulder, right_hip, right_knee)

    # Visualize angle
    cv2.putText(image, 'Left_shoulder: {:.2f}'.format(left_shoulder_angle),
                tuple(np.multiply(left_shoulder, [1280, 720]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA
                )

    cv2.putText(image, 'Right_shoulder: {:.2f}'.format(right_shoulder_angle),
                tuple(np.multiply(right_shoulder, [1280, 720]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA
                )

    cv2.putText(image, 'Left_arm: {:.2f}'.format(left_arm_angle),
                tuple(np.multiply(left_elbow, [1280, 720]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA
                )

    cv2.putText(image, 'Right_arm: {:.2f}'.format(right_arm_angle),
                tuple(np.multiply(right_elbow, [1280, 720]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA
                )

    cv2.putText(image, 'Left_knee: {:.2f}'.format(left_knee_angle),
                tuple(np.multiply(left_knee, [1280, 720]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA
                )

    cv2.putText(image, 'Right_knee: {:.2f}'.format(right_knee_angle),
                tuple(np.multiply(right_knee, [1280, 720]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA
                )

    cv2.putText(image, 'Left_leg: {:.2f}'.format(left_leg_angle),
                tuple(np.multiply(left_hip, [1280, 720]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA
                )

    cv2.putText(image, 'Right_leg: {:.2f}'.format(right_leg_angle),
                tuple(np.multiply(right_hip, [1280, 720]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2, cv2.LINE_AA
                )
    return all_coorddinates

def generate_events(action_def):
    event = {
        "type": current_states['event_type'],
        "action": current_states['action'],
        "schedule": [
            {
                "t": 0,
                "a": "+",
                "pwm": "255",
                "ports": action_def['ports']
            },
            {
                "t": action_def['inflate_duration'],
                "a": "!",
                "pwm": "255",
                "ports": [
                    1,
                    1,
                    1,
                    1,
                    1
                ]
            }
        ]
    }
    return event


def generate_port_events(port_key):
    port_key = port_key- ord('0')
    if port_key == 1:
        ports = [1,0,0,0,0]
    elif port_key == 2:
        ports = [0,1,0,0,0]
    elif port_key == 3:
        ports = [0,0,1,0,0]
    else:
        ports = [1,1,1,1,1]

    event = {
        "type": 'control',
        'action': 'port',
        "schedule": [
            {
                "t":0,
                "a": "+",
                "pwm": "255",
                "ports": ports
            }
        ]
    }
    return event

async def inflate_ports(image, action_def, websocket):
    cmd = action_def['command']
    current_ports = action_def['ports']
    duration= 1.0* action_def['inflate_duration']/1000
    if time.time() > duration + last_cmd_expire_time[cmd]:
        deflate_ports(current_ports, image)
    else:
        cv2.putText(image, f'Inflating Ports : {current_ports} , {duration} s',
            ports_text_position,
            cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA
            )
        if current_states['action'] is not None:
            await websocket.send(json.dumps(generate_events(action_def)))
            current_states['event_type'] = 'tracking'
            current_states['action'] = None

def displaying_deflate(ports_array, image):
    cv2.putText(image, f'Deflating Ports : {ports_array}',
        ports_text_position,
        cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA
    )


async  def execute_action(actions, current_action, image, websocket):

    # draw arrow to show motion direction
    action_def = actions[current_action]
    await inflate_ports(image, action_def, websocket)

def displaying(actions,image, displaying_action, all_coorddinates):
    if displaying_action not in actions:
        cv2.putText(image, f'{displaying_action}',
            ports_text_position,
            cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA
            )
        return
    action_def = actions[displaying_action]
    anchors = get_anchors(action_def, all_coorddinates)
    circle_center = all_coorddinates[action_def["illustration"]["circle_center"]]
    current_ports = action_def["ports"]
    angle = action_def["illustration"]["arrow_angle"]
    direction = action_def["illustration"]["arrow_direction"]
    circle_direction = action_def["illustration"]["circle_direction"]
    end_points = calculate_arrow_endpoints(direction, angle, anchors)
    calculate_circle(image,circle_direction, angle, anchors, circle_center)
    # cv2.arrowedLine(image, end_points[0], end_points[1],
    #     arrow_color, arrow_thickness)
    cmd = action_def['command']
    current_ports = action_def['ports']
    duration= 1.0* action_def['inflate_duration']/1000
    if time.time() > duration*2 + last_cmd_expire_time[cmd]:
        current_states["displaying_action"] = None
    elif time.time() > duration + last_cmd_expire_time[cmd]:
        displaying_deflate(current_ports, image)
    else:
        cv2.putText(image, f'Inflating Ports : {current_ports} , {duration} s',
            ports_text_position,
            cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA
            )

async def remote_cv(websocket):
    print(f"start remote_cv")

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)
    actions = read_configs('actions.json')
    for action, values in actions.items():
        all_cmd.add(values['command'])

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # cv2.putText(image, f'Keys: {all_cmd}',
            #             cmd_text_position,
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 2, cv2.LINE_AA
            #             )
            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                all_coorddinates = visualize_angles(landmarks, mp_pose, image)
                if current_states['action'] is not None and current_states['event_type'] == 'motion':
                    await execute_action(actions, current_states['action'], image, websocket)
                if current_states["displaying_action"] is not None:
                    displaying(actions,image, current_states["displaying_action"], all_coorddinates)
            except AttributeError as err:
                # print(f"Unexpected {err=}, {type(err)=}")
                pass
            except BaseException as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                    )

            cv2.imshow('Iris Demo', image)
            key_pressed = cv2.waitKey(10) & 0xFF
            if key_pressed == ord('q'):
                current_states['action'] = 'quit'
                current_states['event_type'] = 'control'
                event = {
                    "type": 'q',
                }
                await websocket.send(json.dumps(event))
                break
            elif key_pressed == ord(' '):
                current_states['action'] = 'stop'
                current_states['event_type'] = 'control'
                current_states['displaying_action'] = 'stop'
                event = {
                    "type": 'control',
                    'action': 'stop',
                    "schedule": [
                        {
                            "t":0,
                            "a": "!",
                            "pwm": "255",
                            "ports": [
                                1,
                                1,
                                1,
                                1,
                                1
                            ]
                        }
                    ]
                }
                await websocket.send(json.dumps(event))
            elif key_pressed >= ord('1') and key_pressed <= ord('6'):
                current_states['displaying_action'] = 'port control '+str(key_pressed-ord('0'))
                current_states['event_type'] = 'control'
                event = generate_port_events(key_pressed)
                await websocket.send(json.dumps(event))
            elif key_pressed == ord('r'):
                current_states['action'] = 'arm_down_raise'
                current_states['displaying_action'] = 'arm_down_raise'
                current_states['event_type'] = 'motion'
                last_cmd_expire_time['r'] = time.time()
            elif key_pressed == ord('s'):
                current_states['event_type'] = 'motion'
                current_states['action'] = 'arm_bent_straighten'
                current_states['displaying_action'] = 'arm_bent_straighten'
                last_cmd_expire_time['s'] = time.time()

        cap.release()
        cv2.destroyAllWindows()


def my_local_cv(file_path, use_recording=True):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    if use_recording:
        cap= cv2.VideoCapture(file_path)
    else:
        cap = cv2.VideoCapture(0)
    actions = read_configs('actions.json')
    for action, values in actions.items():
        all_cmd.add(values['command'])

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False:
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # cv2.putText(image, f'Keys: {all_cmd}',
            #             cmd_text_position,
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, text_color, 2, cv2.LINE_AA
            #             )
            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                all_coorddinates = visualize_angles(landmarks, mp_pose, image)
                # if current_states['action'] is not None and current_states['event_type'] == 'motion':
                #     await execute_action(actions, current_states['action'], image, websocket)
                # if current_states["displaying_action"] is not None:
                #     displaying(actions,image, current_states["displaying_action"], all_coorddinates)
            except AttributeError as err:
                # print(f"Unexpected {err=}, {type(err)=}")
                pass
            except BaseException as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                    )

            cv2.imshow('Mediapipe Feed', image)
            key_pressed = cv2.waitKey(10) & 0xFF
            if key_pressed == ord('q'):
                current_states['action'] = 'quit'
                current_states['event_type'] = 'control'
                current_states['displaying_action'] = 'quit'
                event = {
                    "type": 'q',
                }
                # await websocket.send(json.dumps(event))
                break
            elif key_pressed == ord(' '):
                current_states['action'] = 'stop'
                current_states['event_type'] = 'control'
                event = {
                    "type": 'control',
                    'action': 'stop',
                    "schedule": [
                        {
                            "t":0,
                            "a": "!",
                            "pwm": "255",
                            "ports": [
                                1,
                                1,
                                1,
                                1,
                                1
                            ]
                        }
                    ]
                }
                # await websocket.send(json.dumps(event))
            elif key_pressed == ord('r'):
                current_states['action'] = 'arm_down_raise'
                current_states['displaying_action'] = 'arm_down_raise'
                current_states['event_type'] = 'motion'
                last_cmd_expire_time['r'] = time.time()
            elif key_pressed == ord('s'):
                current_states['event_type'] = 'motion'
                current_states['action'] = 'arm_bent_straighten'
                current_states['displaying_action'] = 'arm_bent_straighten'
                last_cmd_expire_time['s'] = time.time()

        cap.release()
        cv2.destroyAllWindows()

# def main(argv):

if __name__ == "__main__":
    file_path = sys.argv[1]
    print(file_path)
#    main(sys.argv[1:])
    my_local_cv(file_path)
