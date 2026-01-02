import modal

my_image = modal.Image.from_dockerfile(".")

app = modal.App("fastapi-ml-app")

@app.function(gpu="A10G", image=my_image)  # GPU support here
def run_my_app():
    import subprocess
    subprocess.run(["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "80"])
