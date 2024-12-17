import torch
import cv2
from hubconf import waternet
from PIL import Image
from uiqm_test import calculate_uiqm
import time
# Load the components manually
preprocess, postprocess, model = waternet(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the appropriate device
model = model.to(device)
# Load an example image
image_path = "/home/n/nihita/efficientnet_test/all/10.jpg"
start_time = time.time() 
raw_im = Image.open(image_path)
raw_im = cv2.imread(image_path)
rgb_im = cv2.cvtColor(raw_im, cv2.COLOR_BGR2RGB)
rgb_im = cv2.resize(rgb_im, (720, 480))
# Preprocess the image
rgb_ten, wb_ten, he_ten, gc_ten = preprocess(rgb_im)
rgb_ten = rgb_ten.to(device)
wb_ten = wb_ten.to(device)
he_ten = he_ten.to(device)
gc_ten = gc_ten.to(device)

# Perform inference
with torch.no_grad():
    model_out = model(rgb_ten, wb_ten, he_ten, gc_ten)
end_time = time.time()  # Record the end time

# Postprocess the output
output_image = postprocess(model_out)
import matplotlib.pyplot as plt
if len(output_image.shape) == 4 and output_image.shape[0] == 1:
    output_image = output_image.squeeze(0)  # Removes the batch dimension (N)

# Calculate the elapsed time
inference_time = end_time - start_time
print(f"Inference took {inference_time:.4f} seconds")
# Show the output image
#plt.imshow(output_image)
#save_path = "/home/n/nihita/pp_codes/waternet__/output3.jpg"
#plt.savefig(save_path)  
#plt.show()

# Save or display the output image
#output_path = "/home/n/nihita/pp_codes/waternet__/output4.jpg"

#cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))


#image = output_image.squeeze(axis=0)
uiqm_score = calculate_uiqm(output_image)
print(f'UIQM Score of 3.jpg : {uiqm_score}')
