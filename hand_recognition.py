import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.drawing_utils import draw_landmarks, DrawingSpec, _CONNECTION
from mediapipe.tasks.python.vision.drawing_styles import get_default_hand_connections_style
#Consigue lo puntos de coneccion de mediapipe y los transforma en puntos de coneccion
conn_style = get_default_hand_connections_style()
HAND_CONNECTIONS = [_CONNECTION(a, b) for a, b in conn_style.keys()]

#diseño
white = (255, 255, 255)
landmark_spec = DrawingSpec(color=white, thickness=1, circle_radius=2)
connection_spec = DrawingSpec(color=white, thickness=1, circle_radius=1)

# Si se elimina el hand_landmarker ya o funciona nada
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionTaskRunningMode.VIDEO,  
    num_hands=6,          
    min_hand_detection_confidence=0.5,  
    min_tracking_confidence=0.5,        
)

hand_landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

frame_timestamp = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Espejo horizontal para que se sienta natural (como verse en un espejo)
    frame = cv2.flip(frame, 1)
    # OpenCV usa BGR, MediaPipe necesita RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Envolver el frame como imagen de MediaPipe
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Ejecutar el modelo de IA sobre la imagen
    results = hand_landmarker.detect_for_video(mp_img, frame_timestamp)
    frame_timestamp += 1

    # Dibuja la manos
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # DIbuja los 21 puntos de coneccion en las manos obtenidos
            draw_landmarks(
                frame,
                hand_landmarks,
                HAND_CONNECTIONS,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec,
            )
    # Mostrar el frame en una ventana
    cv2.imshow("hand_rocogizer", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

#finalizar simulacion
cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()
