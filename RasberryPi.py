import time
import os
import json
import smtplib
import numpy as np
import RPi.GPIO as GPIO
import cv2  
import tensorflow as tf
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import paho.mqtt.client as mqtt
import ssl




TRIG = 23  
ECHO = 24  
SERVO_PIN = 18  




EMAIL_ADDRESS = 'lockteamiot@gmail.com'
EMAIL_PASSWORD = 'encrypted password'
RECIPIENT_EMAIL = 'youremail@gmail.com'
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587




GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(SERVO_PIN, GPIO.OUT)




servo = GPIO.PWM(SERVO_PIN, 50) 
servo.start(0)




CONFIDENCE_THRESHOLD = 0.9  
interpreter = tf.lite.Interpreter(model_path="model_unquant2.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def load_labels(label_path="labels2.txt"):
   labels = {}
   with open(label_path, "r") as f:
       for line in f:
           index, name = line.strip().split(' ', 1)
           labels[int(index)] = name
   return labels


labels = load_labels()




def reset_camera(camera_index=0):
   camera = cv2.VideoCapture(camera_index)  
   if not camera.isOpened():
       print("Error: Could not open camera.")
       return None
   return camera




def capture_image(image_path="captured_image.jpg"):
   camera = reset_camera()
   if camera is None:
       return None
   time.sleep(2)  
   ret, frame = camera.read()
   if ret:
       
       brighter_frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=20)  


       cv2.imwrite(image_path, brighter_frame)
       print(f"Image captured and saved at {image_path}")
       camera.release()
       return image_path
   else:
       print("Failed to capture image.")
       camera.release()
       return None


def preprocess_image(image_path):
   image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
   image = tf.keras.preprocessing.image.img_to_array(image)
   image = image / 255.0  # Normalize to [0, 1]
   return np.expand_dims(image, axis=0).astype(np.float32)




def run_inference(image_path):
   image_data = preprocess_image(image_path)
   interpreter.set_tensor(input_details[0]['index'], image_data)
   interpreter.invoke()
   output_data = interpreter.get_tensor(output_details[0]['index'])[0]
   predicted_index = np.argmax(output_data)
   confidence = output_data[predicted_index]
   return (labels[predicted_index] if confidence >= CONFIDENCE_THRESHOLD else "Unknown", confidence)




def send_email(recipient_name, image_path):
   msg = MIMEMultipart()
   msg['From'] = EMAIL_ADDRESS
   msg['To'] = RECIPIENT_EMAIL
   msg['Subject'] = f"{recipient_name} at Your Door" if recipient_name != "Unknown" else "Unknown Person at Your Door"
   email_body = f"{recipient_name} was detected at your door." if recipient_name != "Unknown" else "An unknown individual was detected at your door."
   msg.attach(MIMEText(email_body, 'html'))
   with open(image_path, 'rb') as attachment:
       part = MIMEBase('application', 'octet-stream')
       part.set_payload(attachment.read())
       encoders.encode_base64(part)
       part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
       msg.attach(part)
   server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
   server.starttls()
   server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
   server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())
   server.quit()




def set_servo_angle(angle):
   duty = (angle / 18) + 2
   servo.ChangeDutyCycle(duty)
   time.sleep(0.5)
   servo.ChangeDutyCycle(0)




def get_distance():
   GPIO.output(TRIG, True)
   time.sleep(0.00001)
   GPIO.output(TRIG, False)
   pulse_start, pulse_end = 0, 0
   while GPIO.input(ECHO) == 0:
       pulse_start = time.time()
   while GPIO.input(ECHO) == 1:
       pulse_end = time.time()
   pulse_duration = pulse_end - pulse_start
   return (pulse_duration * 17150) * 0.393701




def on_connect(client, userdata, flags, rc):
   print("Connected to AWS IoT:", rc)
   client.subscribe("raspi/lock")


def on_message(client, userdata, msg):
   command = json.loads(msg.payload.decode()).get("command", "")
   if command == "lock":
       print("Received command: LOCK")
       set_servo_angle(0)
   elif command == "unlock":
       print("Received command: UNLOCK")
       set_servo_angle(180)


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.tls_set(ca_certs="rootCA.pem", certfile="certificate.pem.crt", keyfile="private.pem.key", tls_version=ssl.PROTOCOL_TLSv1_2)
client.tls_insecure_set(True)
client.connect("use the amazon link", 8883, 60)




def monitor_door():
   last_published = None
   start_time = None
   while True:
       distance = get_distance()
       print(f"Distance: {distance:.2f} inches")
       if distance <= 15:
           if start_time is None:
               start_time = time.time()
           elif time.time() - start_time >= 5 and last_published != True:
               image_path = capture_image()
               if image_path:
                   person, confidence = run_inference(image_path)
                   print(f"Detected: {person} with confidence {confidence:.2f}")
                   if person == "Unknown":
                       print("Locking the door.")
                       set_servo_angle(0)
                   else:
                       print(f"Unlocking the door for {person}.")
                       set_servo_angle(180)
                   send_email(person, image_path)
                   last_published = True
       else:
           start_time = None
           last_published = False
       client.publish("raspi/data", json.dumps({"sensorTouched": distance <= 10}), qos=0)
       time.sleep(0.5)


try:
   client.loop_start()
   monitor_door()
except KeyboardInterrupt:
   print("Exiting...")
finally:
   servo.stop()                                                      
   GPIO.cleanup()

