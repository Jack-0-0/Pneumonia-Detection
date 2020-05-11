from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse
from fastai.vision import load_learner, open_image
import tarfile
from io import BytesIO
import os
import sys
import uvicorn

app = Starlette()
path = os.path.dirname(os.path.realpath(__file__))
with tarfile.open("pneu_model_v3.tar.xz") as tar:
    tar.extractall()
file = 'pneu_model_v3'
learn = load_learner(path=path, file=file)


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    img = open_image(BytesIO(bytes))
    prd = learn.predict(img)
    return JSONResponse({"Prediction": str(prd[0])})


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select an X-ray image to be checked for pneumonia:
            <input type="file" name="file">
            <input type="submit" value="*Analyse*">
        </form>
    """)


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=5000)
