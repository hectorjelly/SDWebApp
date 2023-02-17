# Import necessary libraries for Flask web application
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok

# Import PyTorch and the StableDiffusionPipeline class
import torch
from diffusers import StableDiffusionPipeline

# Import libraries for base64 encoding/decoding and reading/writing bytes in memory
import base64
from io import BytesIO

# Load the pre-trained deep learning model using the StableDiffusionPipeline class
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16)

# Set the device for the deep learning model to run on (in this case, a GPU if available)
pipe.to("cuda")

# Create a Flask web application instance
app = Flask(__name__)

# Use the Flask ngrok extension to create a public URL for the web application
run_with_ngrok(app)

# Define a route for the web application to receive HTTP requests
@app.route("/")
def initial():
  # Render the initial HTML template when the root URL is accessed
  return render_template("index.html")

# Define a route for the web application to generate images from user prompts
@app.route('/submit-prompt', methods=['POST'])
def generate_image():
  # Get the prompt input from the HTTP POST request
  prompt = request.form['prompt-input']
  print(f"Generating an image of {prompt}")

  # Generate an image from the prompt using the pre-trained deep learning model
  image = pipe(prompt).images[0]
  print("Image generated! Converting image ...")
  
  # Convert the image to PNG format and encode it in base64
  buffered = BytesIO()
  image.save(buffered, format="PNG")
  img_str = base64.b64encode(buffered.getvalue())
  img_str = "data:image/png;base64," + str(img_str)[2:-1]

  # Return the generated image to the client
  print("Sending image ...")
  return render_template('index.html', generated_image=img_str)

# Run the Flask web application
if __name__ == '__main__':
    app.run()
