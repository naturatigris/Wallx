import cv2
import pygame
from pygame.locals import *
import os
import numpy as np

from FindingBalloons import findBalloons, findContours
from DetectHit import detectHit, showBalloons, findBallInBalloon
import pickle
from calibration import WarpImage

# Initialize Pygame
pygame.init()
screen_width, screen_height = 1400, 700
gameDisplay = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Balloon Game')
points = pickle.load(open("calibration_data.pkl", "rb"))

# Initialize OpenCV capture (assuming webcam)
#cap = cv2.VideoCapture(r"C:\Users\HP\Downloads\tttt.mp4")
#cap = cv2.VideoCapture("http://192.168.220.129:4747/video")
cap = cv2.VideoCapture(0)

image_folder_path = "balloon colours"

white = (255, 255, 255)
black = (0, 0, 0)
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 25)

def load_random_balloon_image():
    balloon_images = [os.path.join(image_folder_path, file) for file in os.listdir(image_folder_path)]
    return pygame.image.load(np.random.choice(balloon_images)).convert_alpha()

balloon_images = [load_random_balloon_image() for _ in range(5)]  # Load images outside the loop

def reset_balloon(i):
    global x, y
    x = np.random.randint(screen_width * 0.2, screen_width * 0.8)
    y = screen_height * 0.6
    return load_random_balloon_image()  # Load a new balloon image

def car(x, y, image):
    width = int(image.get_width() * 5)
    height = int(image.get_height() * 5)
    scaled_image = pygame.transform.scale(image, (width, height))
    gameDisplay.blit(image, (x, y))

def display_score(score):
    score_text = font.render("Score: " + str(score), True, black)
    gameDisplay.blit(score_text, (10, 10))

def display_timer(time_left):
    timer_text = font.render("Time: " + str(time_left), True, black)
    gameDisplay.blit(timer_text, (screen_width - 100, 10))

score = 0
i = 0
balloon_image = reset_balloon(i)  # Initialize balloon position and image
game_end_time = pygame.time.get_ticks() + 60 * 1000  # Set game end time to 60 seconds after current time
game_over = False

running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            # Check if the click is within the balloon area
            if x <= mouse_x <= x + balloon_image.get_width() and y <= mouse_y <= y + balloon_image.get_height():
                score += 1
                balloon_image = reset_balloon(i)  # Reset balloon if clicked

    # Capture frame-by-frame
    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_FPS, 50)

    # Convert OpenCV frame to Pygame surface
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Show video feed in OpenCV window
    cv2.imshow("Video Feed", frame)

    gameDisplay.fill(white)
    car(x, y, balloon_image)  # Show the current balloon image
    display_score(score)

    # Calculate time left
    time_left = max(0, (game_end_time - pygame.time.get_ticks()) // 1000)
    display_timer(time_left)

    y -= 2

    if y < -10:
        i = (i + 1) % len(balloon_images)  # Change index to show different balloon images
        balloon_image = reset_balloon(i)  # Reset balloon and get a new image
    points = pickle.load(open("calibration_data.pkl", "rb"))
    #img=WarpImage(frame,points) 
    img1 = findBalloons(frame)
    img1 = cv2.resize(img1, (0, 0), None, 0.5, 0.5)
    imgContours, bboxs = findContours(img1)
    img2 = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
    cropped_images = detectHit(img2, bboxs)
    if cropped_images:
        score += 1
        balloon_image = reset_balloon(i)  # Reset balloon if hit

    if time_left == 0:
        game_over = True
    if game_over:
        break

    pygame.display.flip()
    clock.tick(40)

# Display final score after the game ends
final_score_display = pygame.display.set_mode((screen_width, screen_height))
final_score_display.fill(white)
final_score_text = font.render("Final Score: " + str(score), True, black)
final_score_display.blit(final_score_text, (screen_width // 2 - 100, screen_height // 2 - 50))
pygame.display.flip()

# Wait for the user to close the final score window
waiting = True
while waiting:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            waiting = False

# Release OpenCV capture and quit Pygame
cap.release()
cv2.destroyAllWindows()
pygame.quit()
