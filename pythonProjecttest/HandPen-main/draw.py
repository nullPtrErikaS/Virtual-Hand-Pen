from tracker import Tracker
import cv2


tracker = Tracker()
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
tracker = Tracker()
while True:
  success, img = cap.read()
  img = cv2.flip(img, 1)
  img = tracker.hand_landmark(img)


  img = tracker.update_drawing_state(img)  # Update the drawing state based on thumb-index distance
  if tracker.results.multi_hand_landmarks:
     hand_landmarks = tracker.results.multi_hand_landmarks[0]
     index_tip = hand_landmarks.landmark[8]  # Index finger tip landmark
     x, y = int(index_tip.x * img.shape[1]), int(index_tip.y * img.shape[0])
     tracker.update_line_segments(x, y)  # Now 'x' and 'y' are defined


  img = tracker.draw(img)


  img = tracker.tracking(img)
  img = tracker.draw(img)
  img = tracker.erase(img)
  img = tracker.change_color(img)
  img = tracker.draw_eraser(img)
  img = tracker.draw_color_boxes(img)




  # cv2.rectangle(img, (tracker.eraser_widget[0], tracker.eraser_widget[1]),
  #               (tracker.eraser_widget[0] + tracker.eraser_widget[2],
  #                tracker.eraser_widget[1] + tracker.eraser_widget[3]), (0, 0, 255), cv2.FILLED)




  cv2.imshow('Drawing board', img)
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()
