import os
import io
import boto3
from celery import Celery, Task
from diffusers import StableDiffusionPipeline
import json

# Get AWS credentials from environment variables
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION")

# IMPORTANT: Replace '172.31.8.113' with the actual private IP of your Redis instance.
redis_ip = "172.31.8.113"
celery_app = Celery("app", broker=f"redis://{redis_ip}:6379/0", backend=f"redis://{redis_ip}:6379/0")

# Use a global variable to cache the model so it's only loaded once per worker process.
global_pipe = None

def get_stable_diffusion_pipeline():
    """
    Initializes and returns a cached Stable Diffusion pipeline.
    This function ensures the model is only loaded once per worker process.
    """
    global global_pipe
    if global_pipe is None:
        try:
            print("Loading Stable Diffusion model...")
            global_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
            global_pipe = global_pipe.to("cuda") # This line requires a GPU and CUDA
            print("Stable Diffusion model loaded successfully.")
        except Exception as e:
            print(f"Error loading Stable Diffusion model: {e}")
            global_pipe = None
    return global_pipe

# Initialize AWS S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

@celery_app.task(name="app.generate_2d_image_task")
def generate_2d_image_task(text_prompt, width, height, num_images, s3_bucket_name, base_filename):
    """
    Generates a 2D image from a text prompt and uploads it to an S3 bucket.
    """
    try:
        pipe = get_stable_diffusion_pipeline()
        if pipe is None:
            return {"status": "error", "message": "Stable Diffusion model failed to load."}

        images = pipe(text_prompt, width=width, height=height, num_images_per_prompt=num_images).images

        results = []
        for i, image in enumerate(images):
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            # Use the images/ folder as the prefix
            s3_filename = f"images/{base_filename}_{i+1}.png"
            s3_client.upload_fileobj(img_bytes, s3_bucket_name, s3_filename)
            print(f"Uploaded {s3_filename} to {s3_bucket_name}")
            results.append(s3_filename)

        return {"status": "success", "result": results}
    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        return {"status": "error", "message": str(e)}

@celery_app.task(name="app.generate_3d_from_2d_task")
def generate_3d_from_2d_task(image_bytes, s3_bucket_name, base_filename):
    """
    Generates a 3D asset from a 2D image.
    """
    try:
        # TODO: Add your 3D generation logic here.
        # This function should take the image_bytes as input and return the 3D model.
        # Example: 3D model data is generated here.

        # Placeholder for the generated 3D model file path.
        # Assume your logic generates a .glb file.
        s3_filename = f"3d_assets/{base_filename}.glb"
        
        # TODO: Replace this with the actual file upload logic.
        # For now, we'll just upload a dummy file to demonstrate the pathing.
        dummy_content = b'This is a placeholder for your 3D model.'
        s3_client.put_object(Bucket=s3_bucket_name, Key=s3_filename, Body=dummy_content)
        
        print(f"Uploaded placeholder 3D asset to {s3_filename}")
        
        # Construct the URL for the uploaded file.
        s3_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{s3_filename}"
        
        return {"status": "success", "result": s3_url}
    except Exception as e:
        print(f"An error occurred during 3D generation: {e}")
        return {"status": "error", "message": str(e)}

@celery_app.task(name="app.decimate_3d_task")
def decimate_3d_task(file_bytes, s3_bucket_name, base_filename):
    """
    Decimates a 3D asset.
    """
    try:
        # TODO: Add your 3D decimation logic here.
        # This function should take the file_bytes of the 3D model as input,
        # decimate it, and return the new model data.
        
        # Placeholder for the decimated 3D model file path.
        s3_filename = f"processed/{base_filename}_decimated.glb"

        # TODO: Replace this with the actual file upload logic.
        # For now, we'll just upload a dummy file.
        dummy_content = b'This is a placeholder for your decimated 3D model.'
        s3_client.put_object(Bucket=s3_bucket_name, Key=s3_filename, Body=dummy_content)
        
        print(f"Uploaded placeholder decimated 3D asset to {s3_filename}")

        s3_url = f"https://{s3_bucket_name}.s3.amazonaws.com/{s3_filename}"

        return {"status": "success", "result": s3_url}
    except Exception as e:
        print(f"An error occurred during decimation: {e}")
        return {"status": "error", "message": str(e)}

@celery_app.task(name="app.generate_image_from_grid_task")
def generate_image_from_grid_task(grid_data_str, width, height, num_images, s3_bucket_name, base_filename):
    """
    Generates an image from a 3D grid.
    """
    # TODO: Implement your image generation from grid logic here.
    # The grid data is provided as a JSON string.
    try:
        grid_data = json.loads(grid_data_str)
        # This is a placeholder for your logic to convert the grid into an image.
        # For now, let's create a simple black and white image.
        from PIL import Image
        img = Image.new('RGB', (width, height), color = 'white')
        
        results = []
        for i in range(num_images):
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            s3_filename = f"images/grid_{base_filename}_{i+1}.png"
            s3_client.upload_fileobj(img_bytes, s3_bucket_name, s3_filename)
            print(f"Uploaded {s3_filename} to {s3_bucket_name}")
            results.append(s3_filename)
            
        return {"status": "success", "result": results}
    except Exception as e:
        print(f"An error occurred during grid visualization: {e}")
        return {"status": "error", "message": str(e)}
