Skip to content
Search or jump to…
Pull requests
Issues
Codespaces
Marketplace
Explore
 
@hectorjelly 
AssemblyAI-Examples
/
flask-gpu-app
Public
Fork your own copy of AssemblyAI-Examples/flask-gpu-app
Code
Issues
Pull requests
Actions
Projects
Security
Insights
flask-gpu-app/main.py /
@ploeber
ploeber initial commit
Latest commit f70fda4 4 days ago
 History
 1 contributor
41 lines (29 sloc)  1.01 KB

from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, request

import torch
from diffusers import StableDiffusionPipeline

import base64
from io import BytesIO

# Load model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

# Start flask app and set to ngrok
app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def initial():
  return render_template('index.html')


@app.route('/submit-prompt', methods=['POST'])
def generate_image():
  prompt = request.form['prompt-input']
  print(f"Generating an image of {prompt}")

  image = pipe(prompt).images[0]
  print("Image generated! Converting image ...")
  
  buffered = BytesIO()
  image.save(buffered, format="PNG")
  img_str = base64.b64encode(buffered.getvalue())
  img_str = "data:image/png;base64," + str(img_str)[2:-1]

  print("Sending image ...")
  return render_template('index.html', generated_image=img_str)


if __name__ == '__main__':
    app.run()
Footer
© 2023 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
flask-gpu-app/main.py at main · AssemblyAI-Examples/flask-gpu-app 
