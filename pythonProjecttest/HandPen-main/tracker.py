import cv2
import mediapipe as mp


class Tracker():
 def __init__(self, static_image_mode=False, max_num_hands=1,
              min_detection_confidence=0.5, min_tracking_confidence=0.5):
     self.static_image_mode = static_image_mode
     self.max_num_hands = max_num_hands
     self.min_detection_confidence = min_detection_confidence
     self.min_tracking_confidence = min_tracking_confidence


     self.hands = mp.solutions.hands.Hands(static_image_mode=self.static_image_mode,
                                           max_num_hands=self.max_num_hands,
                                           min_detection_confidence=self.min_detection_confidence,
                                           min_tracking_confidence=self.min_tracking_confidence)
     self.mpDraw = mp.solutions.drawing_utils
     self.tracking_id = [8, 12]
     self.previous_x = None
     self.previous_y = None
     self.pen_color = (255, 0, 255)  # Default pen color (Magenta)
     self.pen_thickness = 3
     self.tracking_list = []
     # Updated positions
     self.eraser_widget = (100, 10, 75, 75)  # Moving eraser to the top left


     self.color_boxes = [
         (300, 10, 75, 75, (0, 0, 255)),  # Red
         (425, 10, 75, 75, (0, 255, 0)),  # Green
         (550, 10, 75, 75, (255, 0, 0)),  # Blue
         (675, 10, 75, 75, (128, 0, 128)),  # Purple
         (800, 10, 75, 75, (0, 255, 255)),  # Yellow
         (925, 10, 75, 75, (19, 69, 139)),  # Brown
         (1050, 10, 75, 75, (0, 0, 0)),  # Black
     ]


     self.is_drawing = False
     self.line_segments = []  # Each segment: {'start': (x1, y1), 'end': (x2, y2), 'color': (B, G, R)}


 #test
 def update_drawing_state(self, img):
     if self.results.multi_hand_landmarks:
         for hand_landmarks in self.results.multi_hand_landmarks:
             thumb_tip = hand_landmarks.landmark[4]
             index_tip = hand_landmarks.landmark[8]
             x_thumb, y_thumb = int(thumb_tip.x * img.shape[1]), int(thumb_tip.y * img.shape[0])
             x_index, y_index = int(index_tip.x * img.shape[1]), int(index_tip.y * img.shape[0])




             # Calculate the distance between thumb tip and index tip
             distance = ((x_thumb - x_index) ** 2 + (y_thumb - y_index) ** 2) ** 0.5




             # Define a threshold for how close the fingers should be to consider them touching
             touch_threshold = 50  # Adjust based on testing




             # Update the drawing flag based on the distance being greater than the threshold
             self.is_drawing = distance >= touch_threshold
     return img


 def update_line_segments(self, x, y):
     if self.is_drawing and (self.previous_x is not None and self.previous_y is not None):
         # Add a new line segment with the current color
         self.line_segments.append({'start': (self.previous_x, self.previous_y),
                                    'end': (x, y),
                                    'color': self.pen_color})
     # Update the last known position
     self.previous_x, self.previous_y = x, y




 def hand_landmark(self, img):
     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     self.results = self.hands.process(imgRGB)
     if self.results.multi_hand_landmarks:
         for hand_landmarks in self.results.multi_hand_landmarks:
             self.mpDraw.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
     return img




 def tracking(self, img):
     if self.results.multi_hand_landmarks:
         hand_landmarks = self.results.multi_hand_landmarks[0]
         # Let's use the index finger tip for tracking and drawing
         index_tip = hand_landmarks.landmark[8]
         x, y = int(index_tip.x * img.shape[1]), int(index_tip.y * img.shape[0])




         if self.is_drawing:
             if self.previous_x is not None and self.previous_y is not None:
                 # Draw line between previous and current positions
                 cv2.line(img, (self.previous_x, self.previous_y), (x, y), self.pen_color, self.pen_thickness)
                 # Add current position to tracking list if you're using it to redraw paths or for other purposes
                 self.tracking_list.append((x, y))
             # Update previous positions
             self.previous_x, self.previous_y = x, y
         else:
             # If not drawing, just update the previous positions without drawing
             self.previous_x, self.previous_y = x, y




     return img




 def draw(self, img):
     for segment in self.line_segments:
         cv2.line(img, segment['start'], segment['end'], segment['color'], self.pen_thickness)
     return img

 def erase(self, img):
     if self.results.multi_hand_landmarks:
         hand_landmarks = self.results.multi_hand_landmarks[0]
         index_tip = hand_landmarks.landmark[8]
         x_index, y_index = int(index_tip.x * img.shape[1]), int(index_tip.y * img.shape[0])

         # Get the center position of the eraser widget
         eraser_center_x = self.eraser_widget[0] + self.eraser_widget[2] // 2
         eraser_center_y = self.eraser_widget[1] + self.eraser_widget[3] // 2

         # Calculate the distance between index tip and the center of eraser widget
         distance = ((x_index - eraser_center_x) ** 2 + (y_index - eraser_center_y) ** 2) ** 0.5

         # Define a proximity threshold for the eraser (e.g., half the width of the eraser widget)
         eraser_proximity_threshold = self.eraser_widget[2] / 2

         # If the distance is less than the proximity threshold, clear all line segments
         if distance < eraser_proximity_threshold:
             self.line_segments = []  # This clears all drawn lines

     return img

 def change_color(self, img):
     if self.results.multi_hand_landmarks:
         hand_landmarks = self.results.multi_hand_landmarks[0]
         thumb_tip = hand_landmarks.landmark[4]
         middle_tip = hand_landmarks.landmark[12]
         x_thumb, y_thumb = int(thumb_tip.x * img.shape[1]), int(thumb_tip.y * img.shape[0])
         x_middle, y_middle = int(middle_tip.x * img.shape[1]), int(middle_tip.y * img.shape[0])




         # Calculate the distance between thumb tip and middle tip
         distance = ((x_thumb - x_middle) ** 2 + (y_thumb - y_middle) ** 2) ** 0.5




         # Define a threshold for how close the fingers should be to consider them touching
         touch_threshold = 50  # Adjust based on testing




         for id, lm in enumerate(hand_landmarks.landmark):
             x, y = int(lm.x * img.shape[1]), int(lm.y * img.shape[0])
             for box in self.color_boxes:
                 box_x, box_y, box_width, box_height, color = box
                 if distance < touch_threshold and box_x <= x <= box_x + box_width and box_y <= y <= box_y + box_height:
                     self.pen_color = color
                     break
     return img




 def draw_eraser(self, img):
     eraser_x, eraser_y, eraser_width, eraser_height = self.eraser_widget
     cv2.rectangle(img, (eraser_x, eraser_y), (eraser_x + eraser_width, eraser_y + eraser_height), (255, 255, 255),
                   cv2.FILLED)  # Changed color to white
     cv2.putText(img, 'Eraser', (eraser_x + 5, eraser_y + 20),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)  # Changed color to black for text
     return img


 def draw_color_boxes(self, img):
     for box in self.color_boxes:
         box_x, box_y, box_width, box_height, color = box
         cv2.rectangle(img, (box_x, box_y), (box_x + box_width, box_y + box_height), color, cv2.FILLED)
     return img