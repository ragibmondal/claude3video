import anthropic
import base64
import cv2
from dotenv import load_dotenv
import os

# Get API key
api_key = os.getenv("ANTHROPIC_API_KEY")

if api_key is None:
     # If the API key is not in the environment variable, read it from the .env file
     load_dotenv()
     api_key = os.getenv("ANTHROPIC_API_KEY")

if api_key is None:
     # If the API key is not found in the environment variable or .env file, display an error message and exit
     print("Error: ANTHROPIC_API_KEY not found in environment variables or .env file.")
     exit(1)

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=api_key)

def get_frames_from_video(file_path, max_images=20):
     video = cv2.VideoCapture(file_path)
     base64_frames = []
     while video.isOpened():
         success, frame = video.read()
         if not success:
             break
         _, buffer = cv2.imencode(".jpg", frame)
         base64_frame = base64.b64encode(buffer).decode("utf-8")
         base64_frames.append(base64_frame)
     video.release()

     # Limit the number of images to select
     selected_frames = base64_frames[0::len(base64_frames)//max_images][:max_images]

     return selected_frames, buffer

def get_text_from_video(file_path, prompt, model, max_images=20):
     # Get frames from the video and encode them to base64
     print(f"{file_path}:\nStart acquiring frames")
     base64_frames, buffer = get_frames_from_video(file_path, max_images)
     print("Frame acquisition complete")
     # Send a request to Claude API
     with client.messages.stream(
         model=model, # model specification
         max_tokens=1024, # Maximum number of tokens
         messages=[
             {
                 "role": "user",
                 "content": [
                     *map(lambda x: {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": x}}, base64_frames),
                     {
                         "type": "text",
                         "text": prompt
                     }
                 ],
             }
         ],
     ) as stream:
         for text in stream.text_stream:
             print(text, end="", flush=True)

if __name__ == "__main__":
     video_file_path = os.path.join("resources", "video_name.mp4") # Specify the video file path
     prompt = "This is a frame image of a video. Please differentiate the flow and action of the video from the beginning to the end and explain it in Japanese." # Specify the prompt
     model = "claude-3-sonnet-20240229" # Specify model "claude-3-opus-20240229" or "claude-3-sonnet-20240229"

     get_text_from_video(video_file_path, prompt, model)
