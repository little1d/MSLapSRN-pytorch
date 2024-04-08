"""处理使用swanlab记录Image的逻辑，可能后面合并到train.py中"""

from train import run
import swanlab

# lr
lr_image_list = []
for i in range(5):
    image_path = "./dataset/" + str(i) + "_lr.jpg"
    image = swanlab.Image(image_path, caption=f"lr image {i}")
    lr_image_list.append(image)

# 2x
image_list_2x = []
for i in range(5):
    image_path = "./demo_image_output/" + str(i) + "_out_2x.jpg"
    image = swanlab.Image(image_path, caption=f"out_2x image {i}")
    image_list_2x.append(image)

# 4x
image_list_4x = []
for i in range(5):
    image_path = "./demo_image_output/" + str(i) + "_out_2x.jpg"
    image = swanlab.Image(image_path, caption=f"out_4x image {i}")
    lr_image_list.append(image)

# image_path = "./out_2x.png"
# image = swanlab.Image(image_path)
run.log({"out_2x": image})
