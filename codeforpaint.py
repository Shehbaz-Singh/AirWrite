import cv2
import numpy as np
import os
import HandTrackingModule as htm
from flask import Blueprint, render_template
from tensorflow.keras.models import load_model
import keyboard
import pygame
import time


# Create the Blueprint for the VirtualPainter
VirtualPainter = Blueprint("HandTrackingModule", __name__, static_folder="static", template_folder="templates")

@VirtualPainter.route("/feature")
def start_painter():
    # Color settings
    color_white = (255, 255, 255)
    color_black = (0, 0, 0)
    color_red = (0, 0, 255)
    color_yellow = (0, 255, 255)
    color_green = (0, 255, 0)
    bg_color = (255, 255, 255)
    fg_color = (0, 255, 0)
    border_color = (0, 255, 0)
    prev_draw_color = (0, 0, 1)
    active_color = (0, 0, 255)
    boundary_increase = 5

    # Video capture and canvas settings
    capture = cv2.VideoCapture(0)
    frame_width, frame_height = 1280, 720
    capture.set(3, frame_width)
    capture.set(4, frame_height)
    canvas_image = np.zeros((frame_height, frame_width, 3), np.uint8)

    # Initialize pygame for additional display
    pygame.init()
    font = pygame.font.SysFont('freesansbold.tff', 18)
    display_surface = pygame.display.set_mode((frame_width, frame_height), flags=pygame.HIDDEN)
    pygame.display.set_caption("Digit Recognition Board")
    draw_x_coords = []
    draw_y_coords = []

    # Load header images
    header_folder = "header"
    header_files = os.listdir(header_folder)
    header_images = [cv2.imread(f'{header_folder}/{img_file}') for img_file in header_files]
    active_header = header_images[0]

    # Load trained models
    prediction_label = ""
    prediction_mode = "off"
    alpha_model = load_model("Model1.h5")
    num_model = load_model("Model2.h5")
    alpha_labels = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
                    10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't',
                    20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: ''}
    num_labels = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
    rect_start_x, rect_end_x = 0, 0
    rect_start_y, rect_end_y = 0, 0

    # Initialize hand detector
    hand_detector = htm.handDetector(detectionCon=0.85)
    x_prev, y_prev = 0, 0
    brush_size = 15
    eraser_size = 30
    recognition_mode = "OFF"
    recognition_color = color_red

    # Main drawing loop
    while True:
        success, frame = capture.read()
        frame = cv2.flip(frame, 1)
        frame = hand_detector.findHands(frame)
        landmarks = hand_detector.findPosition(frame, draw=False)
        if hand_detector.isPalmOpen():
            canvas_image = np.zeros((frame_height, frame_width, 3), np.uint8)  # Clear the canvas

        
        text_color1 = (105, 105, 105)  # Dark grey
        text_color = (0, 0, 0)
         
        # Updated placement to the bottom-left corner
        cv2.putText(frame, "Press 1 for Alphabet Recognition Mode", (10, 700), 3, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "Press 2 for Number Recognition Mode", (10, 685), 3, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "Press 3 to Turn Off Recognition Mode", (10, 670), 3, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "Press 4 to Exit", (10, 655), 3, 0.5, text_color, 1, cv2.LINE_AA)  # Added exit instruction
        cv2.putText(frame, f"RECOGNITION MODE: {recognition_mode}", (10, 640), 3, 0.5, recognition_color, 1, cv2.LINE_AA)



        if keyboard.is_pressed('1'):
            if prediction_mode != "alpha":
                prediction_mode = "alpha"
                recognition_mode, recognition_color = "ALPHABETS", text_color1

        if keyboard.is_pressed('2'):
            if prediction_mode != "num":
                prediction_mode = "num"
                recognition_mode, recognition_color = "NUMBER", text_color1

        if keyboard.is_pressed('3'):
            if prediction_mode != "off":
                prediction_mode = "off"
                recognition_mode, recognition_color = "OFF", text_color1
                
        if keyboard.is_pressed('4'):
            capture.release()
            cv2.destroyAllWindows()

            x_prev, y_prev = 0, 0
            prediction_label = ""
            rect_start_x, rect_end_x = 0, 0
            rect_start_y, rect_end_y = 0, 0
            draw_x_coords = []
            draw_y_coords = []
            time.sleep(0.5)

        if len(landmarks) > 0:
            index_x, index_y = landmarks[8][1:]
            middle_x, middle_y = landmarks[12][1:]

            fingers_up = hand_detector.fingersUp()

            if fingers_up[1] and fingers_up[2]:
                draw_x_coords = sorted(draw_x_coords)
                draw_y_coords = sorted(draw_y_coords)

                if len(draw_x_coords) > 0 and len(draw_y_coords) > 0 and prediction_mode != "off":
                    if active_color != (0, 0, 0) and prev_draw_color != (0, 0, 0):
                        rect_start_x, rect_end_x = max(draw_x_coords[0] - boundary_increase, 0), min(frame_width, draw_x_coords[-1] + boundary_increase)
                        rect_start_y, rect_end_y = max(0, draw_y_coords[0] - boundary_increase), min(draw_y_coords[-1] + boundary_increase, frame_height)
                        draw_x_coords = []
                        draw_y_coords = []

                        cropped_img = np.array(pygame.PixelArray(display_surface))[rect_start_x:rect_end_x, rect_start_y:rect_end_y].T.astype(np.float32)

                        cv2.rectangle(canvas_image, (rect_start_x, rect_start_y), (rect_end_x, rect_end_y), border_color, 3)
                        resized_img = cv2.resize(cropped_img, (28, 28))
                        resized_img = np.pad(resized_img, (10, 10), 'constant', constant_values=0)
                        resized_img = cv2.resize(resized_img, (28, 28)) / 255

                        if prediction_mode == "alpha":
                            prediction_label = str(alpha_labels[np.argmax(alpha_model.predict(resized_img.reshape(1, 28, 28, 1)))])
                        if prediction_mode == "num":
                            prediction_label = str(num_labels[np.argmax(num_model.predict(resized_img.reshape(1, 28, 28, 1)))])

                        pygame.draw.rect(display_surface, color_black, (0, 0, frame_width, frame_height))
                        cv2.rectangle(canvas_image, (rect_start_x + 50, rect_start_y - 20), (rect_start_x, rect_start_y), bg_color, -1)
                        cv2.putText(canvas_image, prediction_label, (rect_start_x, rect_start_y - 5), 3, 0.5, fg_color, 1, cv2.LINE_AA)
                    else:
                        draw_x_coords = []
                        draw_y_coords = []

                x_prev, y_prev = 0, 0
                if index_x > 1148:  # Right-side interaction area
                    prev_draw_color = active_color
    
    # Clear canvas area at the top right
                    if 50 < index_y < 150:  # Clear Canvas Area (200px)
                        canvas_image = np.zeros((frame_height, frame_width, 3), np.uint8)  # Clear canvas
    
    # Change marker to red
                    elif 150 < index_y < 250:  # Change Marker to Red Area (200px)
                        active_header = header_images[0]
                        active_color = color_red
    
    # Change marker to yellow
                    elif 250 < index_y < 350:  # Change Marker to Yellow Area (200px)
                        active_header = header_images[1]
                        active_color = color_yellow
    
    # Change marker to green
                    elif 350 < index_y < 850:  # Change Marker to Green Area (200px)
                        active_header = header_images[2]
                        active_color = color_green
        
    # Clear canvas again at the bottom
                    elif 850 < index_y < 1050:  # Clear Canvas Area Again (200px)
                        canvas_image = np.zeros((frame_height, frame_width, 3), np.uint8)  # Clear canvas


                cv2.rectangle(frame, (index_x, index_y - 25), (middle_x, middle_y + 25), active_color, cv2.FILLED)

            elif fingers_up[1] and not fingers_up[2]:
                draw_x_coords.append(index_x)
                draw_y_coords.append(index_y)

                cv2.circle(frame, (index_x, index_y - 15), 15, active_color, cv2.FILLED)
                if x_prev == 0 and y_prev == 0:
                    x_prev, y_prev = index_x, index_y

                if active_color == (0, 0, 0):
                    cv2.line(frame, (x_prev, y_prev), (index_x, index_y), active_color, eraser_size)
                    cv2.line(canvas_image, (x_prev, y_prev), (index_x, index_y), active_color, eraser_size)
                else:
                    cv2.line(frame, (x_prev, y_prev), (index_x, index_y), active_color, brush_size)
                    cv2.line(canvas_image, (x_prev, y_prev), (index_x, index_y), active_color, brush_size)
                    pygame.draw.line(display_surface, color_white, (x_prev, y_prev), (index_x, index_y), brush_size)

                x_prev, y_prev = index_x, index_y
            else:
                x_prev, y_prev = 0, 0

        gray_img = cv2.cvtColor(canvas_image, cv2.COLOR_BGR2GRAY)
        _, inverted_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY_INV)
        inverted_img = cv2.cvtColor(inverted_img, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, inverted_img)
        frame = cv2.bitwise_or(frame, canvas_image)

        active_header = cv2.resize(active_header, (132, 720))
        frame[0:720, 1148:1280] = active_header
        pygame.display.update()
        cv2.imshow("Drawing App", frame)
        cv2.waitKey(1)

# Start the drawing application
start_painter()