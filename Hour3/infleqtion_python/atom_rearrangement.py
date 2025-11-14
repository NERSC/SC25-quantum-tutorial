from __future__ import annotations

try:
    import general_superstaq as gss
    import numpy as np
    from PIL import Image
except ImportError:
    print("Installing required packages...")
    #pip install -r requirements.txt
    print("You may need to restart the kernel to import newly installed packages.")
    import general_superstaq as gss
    import numpy as np
    from PIL import Image

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

# Load an image of your choice
image_path = "heart_logo.png"
image = Image.open(image_path)

# Resize to an API supported size
image = image.resize((7, 7))

# Convert to grayscale
image = image.convert("L")

threshold = 180  # Adjust if needed to acheive sufficient binary encoding
bitmap_array = 1 - (np.array(image) // threshold)

print(bitmap_array)

def ascii_preview(arr: npt.NDArray) -> str:
    choices = sorted(np.unique(arr).tolist())
    choice_map = {choices[i]: " .#"[i] for i in range(len(choices))}
    rows = [" ".join(choice_map[v] for v in row) for row in arr]
    return "\n".join(rows)

print(ascii_preview(bitmap_array))

api_key = "your_api_key"
superstaq_service = gss.Service(api_key=api_key)

# Pass your bitmap array to the `submit_atom_picture` endpoint:
request_id = superstaq_service.submit_atom_picture(bitmap_array)

print(request_id)
