import os

from pathlib import Path

from dtoolbioimage import Image


LOG_PATH = "imagelog"

count = 0

def info(im, caption):
    global count
    
    Path(LOG_PATH).mkdir(exist_ok=True, parents=True)

    fname = f"{count}_{caption}.png"
    output_fpath = os.path.join(LOG_PATH, fname)

    if hasattr(im, "save"):
        im.save(output_fpath)
    else:
        im.view(Image).save(output_fpath)

    count += 1
