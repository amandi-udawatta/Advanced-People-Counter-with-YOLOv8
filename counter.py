import cv2
import imutils
from ultralytics import YOLO

capture = cv2.VideoCapture('people_sample.mp4')

#load yolov8 model
#this is pretrained using COCO dataset
#yolov8n for nano, yolov8s for small
model = YOLO('yolov8n.pt')

# position of the counting line (ROI) in px
roi_line_position = 300 # when you increase this, roi line moves down

people_in=0
people_out=0


# Loop over frames from the video
while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)

    #detect objects using yolov8 and get results back
    results = model(frame)

    #line(video frame we are drawing on, start coordinate, end coordinate, colour of line, thickness of ine in px)
    #frame.shape[1] gives the width of the frame(rightmost edge of the frame)
    cv2.line(frame, (0, roi_line_position), (frame.shape[1], roi_line_position), (0,255,255), 2)
    
    boxes = results[0].boxes

    for box in boxes:

        #In COCO dataset, 'person' is in 0 class. therefore we skip non-persons here
        if box.cls[0] != 0:
            continue

        # Check if the confidence score meets the threshold
        if box.conf[0] < 0.5:
            continue

        #get bounding box for contoured areas
        x, y, w, h = box.xyxy[0].int().tolist()
        center_y = (y+h) // 2

        #draw bounding box
        #rectangle(image, lower left, uppser right, colour, thickness)
        cv2.rectangle(frame, (x,y), (x+w , y+h), (0,255,0), 2)

        #check if going in or out using roi
        if center_y > roi_line_position - 8 and center_y < roi_line_position + 8 : #to check if a person is crossing the line
                if center_y < roi_line_position:
                    people_out += 1
                else:
                    people_in += 1

    # Display the counts on the frame
    #puttext(image, text, position, font, font size, color, thickness)
    cv2.putText(frame, f"In: {people_in}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Out: {people_out}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('YOLOv8 people counter', frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
capture.release()
cv2.destroyAllWindows()

