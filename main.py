import cv2
from ultralytics import YOLO
import math
import time
import smtplib
from email.message import EmailMessage
import imutils
from io import BytesIO



# class names for model
class_names1 = ['Non Violent', 'Violence']

# coco class_names for model2
class_names = [
    "Person", "Bicycle", "Car", "Motorbike", "Aeroplane", "Bus", "Train", "Truck",
    "Boat", "Traffic Light", "Fire Hydrant", "Stop Sign", "Parking Meter", "Bench",
    "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra",
    "Giraffe", "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase", "Frisbee", "Skis",
    "Snowboard", "Sports Ball", "Kite", "Baseball Bat", "Baseball Glove", "Skateboard",
    "Surfboard", "Tennis Racket", "Bottle", "Wine Glass", "Cup", "Fork", "Knife",
    "Spoon", "Bowl", "Banana", "Apple", "Sandwich", "Orange", "Broccoli", "Carrot",
    "Hot Dog", "Pizza", "Donut", "Cake", "Chair", "Sofa", "Potted Plant", "Bed",
    "Dining Table", "Toilet", "TV Monitor", "Laptop", "Mouse", "Remote", "Keyboard",
    "Cell Phone", "Microwave", "Oven", "Toaster", "Sink", "Refrigerator", "Book",
    "Clock", "Vase", "Scissors", "Teddy Bear", "Hair Dryer", "Toothbrush"
]

# Email configuration
smtp_server = 'smtp.gmail.com'
smtp_port = 587  # Port for SMTP (commonly 587 for TLS)
smtp_username = os.environ('USER_EMAIL')
smtp_password = os.environ('PASSWORD')
from_email = os.environ('USER_EMAIL')
to_email = os.environ('TO_SENDER_EMAIL')
subject = 'Frame from Fight Alert System'

# Function to send email
def send_frame_as_email(frame, to_email):
    """SMTP ALERT SYSTEM"""
    # Encode frame to JPEG format
    is_success, buffer = cv2.imencode(".jpg", frame)
    io_buf = BytesIO(buffer)

    # Create the email message
    msg = EmailMessage()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.set_content('Fight Detected. Please Check Out the frames: ')

    # Attach the frame to the email
    io_buf.seek(0)
    msg.add_attachment(io_buf.read(), maintype='image', subtype='jpeg', filename='frame.jpg')

    # Connect to the SMTP server and send the email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()  # Start TLS encryption
        server.login(smtp_username, smtp_password)
        server.send_message(msg)
        print(f'Email sent to {to_email}')


# Initialize webcam capture
cap = cv2.VideoCapture('/Users/boss/Desktop/yolo/test.mp4')
cap.set(3, 800)  # Set width
cap.set(4, 720)  # Set height

# Load the YOLO model with pre-trained weights
model = YOLO('/Users/boss/Desktop/yolo/best.pt') # model for violence detection
model2 = YOLO('/Users/boss/Desktop/yolov8n.pt') # model for person/object detection

# Initialize a counter for the "Violence" class
violence_counter = 0

# Loop to process webcam frames
while True:
    success, img = cap.read()
    results = model2(img, stream=True)
    people_count = 0

    # Iterate over detection results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get bounding box coordinates and cast them to integers
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1


            # Calculate confidence
            conf = math.ceil(box.conf[0] * 100) / 100

            # Get class identification
            cls = box.cls[0]
            class_name = class_names[int(cls)]


            if class_name == 'Person' and conf > 0.5:
                # Put class name and confidence on the image
                # Draw bounding box in yellow color
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                # Put class name and confidence on the frame in small text size
                cv2.putText(img, f'{class_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1)

                people_count += 1

                if people_count >= 2:
                    results_2 = model.predict(img)

                    for r in results_2:
                        boxes = r.boxes
                        print(results_2)
                        for box in boxes:
                            # Get bounding box coordinates and cast them to integers
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            w, h = x2 - x1, y2 - y1

                            # Calculate confidence
                            conf = math.ceil(box.conf[0] * 100) / 100

                            # Get class identification
                            cls = box.cls[0]
                            class_name = class_names1[int(cls)]
                            if class_name == 'Violence':
                                violence_counter += 1
                            else:
                                violence_counter = 0

                            cv2.putText(img, f'{class_name} {conf}',(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # no box just text will appear when
                            # violence is detected

                            # If "Violence" has been detected for 100 consecutive frames, print the message
                            if violence_counter >= 10:
                                print("Violence detected consistently for 10 frames.")
                                time.sleep(0.5)
                                print('Initialising alert system...')
                                time.sleep(0.5)
                                print('Alerting Authorities')
                                frame = imutils.resize(img, width=600)  # Resize for easier emailing
                                send_frame_as_email(frame, to_email) # sending email if violence is detected frames 

    # Show the processed image
    cv2.imshow('Violence Detection', img)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
