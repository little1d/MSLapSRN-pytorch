"""处理使用swanlab记录Image的逻辑，可能后面合并到train.py中"""

from train import run
import swanlab

# lr
lr_image_list = []
for i in range(2):
    image_path = "./dataset/" + str(i) + "_lr.png"
    image = swanlab.Image(image_path, caption=f"lr image {i}", size=128)
    lr_image_list.append(image)

# 2x
image_list_2x = []
for i in range(2):
    image_path = "./demo_image_output/" + str(i) + "_out_2x.png"
    image = swanlab.Image(image_path, caption=f"out_2x image {i}", size=256)
    image_list_2x.append(image)

# 4x
image_list_4x = []
for i in range(2):
    image_path = "./demo_image_output/" + str(i) + "_out_2x.png"
    image = swanlab.Image(image_path, caption=f"out_4x image {i}", size=512)
    image_list_4x.append(image)

run.log({"lr": lr_image_list, "out_2x": image_list_2x, "out_4x": image_list_4x})
