import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import numpy as np
import os
import boto3
import logging
import datetime
from botocore.exceptions import ClientError
import trimesh
from uuid import uuid4
import json
import tempfile
from celery import Celery
import io

# Initialize Celery with embedded configuration
# The name 'app' is used here to match your intended Celery file name.
celery_app = Celery('app')
celery_app.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/0',
    result_serializer='json',
    task_imports=('app',),  # Important: The task module is now 'app'
    task_queues={
        'default': {
            'exchange': 'default',
            'binding_key': 'default',
        },
    }
)

logging.basicConfig(level=logging.INFO)

def upload_file_to_s3(file_data, bucket, object_name=None):
    """
    Uploads file data to an S3 bucket from a BytesIO object.
    """
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_fileobj(file_data, bucket, object_name)
        logging.info(f"Successfully uploaded to s3://{bucket}/{object_name}")
        return f"https://{bucket}.s3.amazonaws.com/{object_name}"
    except ClientError as e:
        logging.error(f"Failed to upload to S3: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during S3 upload: {e}")
        return None

def decimate_mesh(mesh_data, target_faces=1000):
    """
    Decimates a 3D mesh from a file-like object.
    """
    print(f"\n--- Starting Decimation Step ---")
    print(f"Decimating mesh to approximately {target_faces} faces...")
    
    mesh = trimesh.load(file_obj=io.BytesIO(mesh_data), file_type='glb')
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            print("Warning: Scene contains no geometry. Decimation skipped.")
            return None
        if mesh.dump():
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        else:
            print("Warning: Scene has no dumpable geometry. Decimation skipped.")
            return None

    decimated_mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
    print(f"Original faces: {len(mesh.faces)}, Decimated faces: {len(decimated_mesh.faces)}")
    
    decimated_data = io.BytesIO()
    decimated_mesh.export(file_obj=decimated_data, file_type='glb')
    decimated_data.seek(0)
    return decimated_data

@celery_app.task(bind=True)
def generate_2d_image_task(self, text_prompt: str, width: int, height: int, num_images: int, s3_bucket_name: str, base_filename: str):
    """
    Celery task to generate 2D images from a text prompt.
    The filename includes the base filename, a timestamp, and an index.
    """
    self.update_state(state='PROGRESS', meta={'status': "Optimizing prompt and generating image(s)...", 'result': []})
    
    optimized_prompt = f"{text_prompt}, 3D render, octane render, unreal engine, cinematic, highly detailed"
    
    stable_diffusion_model_id = "runwayml/stable-diffusion-v1-5"
    
    # Try to use GPU, fallback to CPU
    try:
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            stable_diffusion_model_id, torch_dtype=torch.float16
        ).to("cuda")
    except Exception:
        sd_pipe = StableDiffusionPipeline.from_pretrained(stable_diffusion_model_id).to("cpu")

    negative_prompt = "blurry, low quality, bad anatomy, ugly, artifacts, text, watermark"
    
    generated_urls = []
    for i in range(num_images):
        current_status = f"Generating image {i+1}/{num_images}..."
        self.update_state(state='PROGRESS', meta={'status': current_status, 'result': generated_urls})
        
        image = sd_pipe(
            optimized_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,
            width=width,
            height=height
        ).images[0]
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"images/{base_filename}_{timestamp}_{i}.png"
        
        image_data = io.BytesIO()
        image.save(image_data, format='PNG')
        image_data.seek(0)
        
        s3_url = upload_file_to_s3(image_data, s3_bucket_name, filename)
        if s3_url:
            print(f"Uploaded 2D image to: {s3_url}")
            generated_urls.append(s3_url)
    
    self.update_state(state='SUCCESS', meta={'status': "Image generation complete!", 'result': generated_urls})
    return generated_urls

@celery_app.task(bind=True)
def generate_image_from_grid_task(self, grid_data_str: str, width: int, height: int, num_images: int, s3_bucket_name: str, base_filename: str):
    """
    Celery task to generate image visualizations from grid data.
    The filename includes the base filename, a timestamp, and an index.
    """
    self.update_state(state='PROGRESS', meta={'status': "Parsing grid data and generating visualization...", 'result': []})

    try:
        grid_data = json.loads(grid_data_str)
        # Validation checks
        if not isinstance(grid_data, list) or not all(isinstance(row, list) and all(isinstance(cell, int) for cell in row) for row in grid_data):
            raise ValueError("Grid data must be a valid JSON array of arrays of integers.")
        if not grid_data or not grid_data[0]:
            raise ValueError("Grid data is empty.")

        cell_size = 50 
        img_width = len(grid_data[0]) * cell_size
        img_height = len(grid_data) * cell_size

        terrain_colors = {
            0: (144, 238, 144), 1: (34, 139, 34), 2: (139, 69, 19), 3: (65, 105, 225),
            4: (244, 164, 96), 5: (240, 248, 255), 6: (102, 51, 153), 7: (160, 82, 45),
            8: (105, 105, 105), 9: (128, 128, 128)
        }

        generated_urls = []
        for i in range(num_images):
            img = Image.new('RGB', (img_width, img_height), color='white')
            d = ImageDraw.Draw(img)

            for r_idx, row in enumerate(grid_data):
                for c_idx, cell_value in enumerate(row):
                    color = terrain_colors.get(cell_value, (0, 0, 0))
                    x1, y1 = c_idx * cell_size, r_idx * cell_size
                    x2, y2 = x1 + cell_size, y1 + cell_size
                    d.rectangle([x1, y1, x2, y2], fill=color, outline=(0,0,0))
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"images/grid_visualizations/{base_filename}_{timestamp}_{i}.png"
            
            image_data = io.BytesIO()
            img.save(image_data, format='PNG')
            image_data.seek(0)
            
            s3_url = upload_file_to_s3(image_data, s3_bucket_name, filename)
            if s3_url:
                print(f"Uploaded grid visualization to: {s3_url}")
                generated_urls.append(s3_url)
            
            self.update_state(state='PROGRESS', meta={'status': f"Generated visualization {i+1}/{num_images}", 'result': generated_urls})

        self.update_state(state='SUCCESS', meta={'status': "Grid visualization complete!", 'result': generated_urls})
        return generated_urls

    except json.JSONDecodeError:
        self.update_state(state='FAILURE', meta={'error': "Invalid JSON format for Grid Data."})
        raise
    except ValueError as ve:
        self.update_state(state='FAILURE', meta={'error': str(ve)})
        raise
    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': f"An unexpected error occurred: {e}"})
        raise

@celery_app.task(bind=True)
def generate_3d_from_2d_task(self, image_bytes: bytes, s3_bucket_name: str, base_filename: str):
    """
    Celery task to generate a 3D model from a 2D image.
    The filename is the base filename provided by the user.
    """
    self.update_state(state='PROGRESS', meta={'status': "Generating 3D model from 2D image...", 'result': None})
    
    image_2d_input = Image.open(io.BytesIO(image_bytes))
    
    hunyuan_model_id = 'tencent/Hunyuan3D-2mini'
    hunyuan_subfolder = 'hunyuan3d-dit-v2-mini'
    
    hunyuan_pipeline = None
    try:
        hunyuan_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            hunyuan_model_id,
            subfolder=hunyuan_subfolder,
            use_safetensors=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except Exception as e:
        print(f"Failed to load Hunyuan3D-2mini with GPU. Error: {e}")
        try:
            hunyuan_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                hunyuan_model_id,
                subfolder=hunyuan_subfolder,
                use_safetensors=True
            ).to("cpu")
            print("Successfully loaded model on CPU.")
        except Exception as cpu_e:
            error_msg = f"Failed to load Hunyuan3D-2mini on CPU. Error: {cpu_e}"
            print(error_msg)
            self.update_state(state='FAILURE', meta={'error': error_msg})
            return
            
    with torch.no_grad():
        mesh = hunyuan_pipeline(
            image=image_2d_input,
            num_inference_steps=30,
            octree_resolution=256,
            generator=torch.Generator(device=hunyuan_pipeline.device).manual_seed(42)
        )[0]
    
    mesh_data = io.BytesIO()
    mesh.export(file_obj=mesh_data, file_type='glb')
    mesh_data.seek(0)
    
    filename = f"3d_assets/{base_filename}.glb"
    s3_url = upload_file_to_s3(mesh_data, s3_bucket_name, filename)
    
    if s3_url:
        self.update_state(state='SUCCESS', meta={'status': "3D model generation complete and uploaded to S3!", 'result': s3_url})
    else:
        self.update_state(state='FAILURE', meta={'error': "Failed to upload 3D model to S3."})

@celery_app.task(bind=True)
def decimate_3d_task(self, input_3d_bytes: bytes, s3_bucket_name: str, base_filename: str):
    """
    Celery task to decimate a 3D model.
    The filename is the base filename with "_decimated" appended.
    """
    self.update_state(state='PROGRESS', meta={'status': "Decimating 3D model...", 'result': None})
    
    try:
        decimated_mesh_data = decimate_mesh(input_3d_bytes, target_faces=1000)
        
        if decimated_mesh_data is not None:
            filename = f"processed/{base_filename}_decimated.glb"
            s3_url = upload_file_to_s3(decimated_mesh_data, s3_bucket_name, filename)
            if s3_url:
                self.update_state(state='SUCCESS', meta={'status': "3D model decimated and uploaded to S3!", 'result': s3_url})
            else:
                self.update_state(state='FAILURE', meta={'error': "Failed to upload decimated 3D model to S3."})
        else:
            self.update_state(state='FAILURE', meta={'error': "Decimation failed: No valid mesh produced."})
            
    except Exception as e:
        error_msg = f"Error during decimation: {e}"
        self.update_state(state='FAILURE', meta={'error': error_msg})
        raise
