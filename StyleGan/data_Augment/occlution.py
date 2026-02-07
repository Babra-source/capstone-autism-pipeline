import torch
from pathlib import Path
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


# Step 1: Resize and convert images to tensors
simple_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Step 2: Make a black box in the center of images
class BlackBox:
    def __init__(self):
        self.box_size_min = 0.2  # 20% of image
        self.box_size_max = 0.55  # 40% of image
    
    def __call__(self, image):
        # Get image size
        channels, height, width = image.shape
        
        # Decide random box size
        box_area = torch.rand(1).item() * (self.box_size_max - self.box_size_min) + self.box_size_min
        box_height = int(height * box_area)
        box_width = int(width * box_area)
        
        # Put box in center
        center_y = height // 2
        center_x = width // 2
        
        top = center_y - box_height // 2
        bottom = top + box_height
        left = center_x - box_width // 2
        right = left + box_width
        
        # Make it black
        image[:, top:bottom, left:right] = 0
        
        return image


# Step 3: Set up paths
dataset_path = "C:/Users/HomePC/Capstone_code/capstone-autism-pipeline/data/Autism_Dataset/train"
output_path = "C:/Users/HomePC/Capstone_code/capstone-autism-pipeline/data/occluded_images"

# Step 4: Load images
dataset = datasets.ImageFolder(dataset_path, transform=simple_transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Step 5: Create output folders
Path(output_path).mkdir(parents=True, exist_ok=True)
for class_name in dataset.classes:
    Path(output_path, class_name).mkdir(exist_ok=True)

# Step 6: Add black boxes and save
black_box = BlackBox()
image_count = 0

for images, labels in loader:
    for i in range(len(images)):
        # Add black box
        image_with_box = black_box(images[i])
        
        # Figure out which folder to save to
        label = labels[i].item()
        class_name = dataset.classes[label]
        
        # Save the image
        filename = f"occluded_{image_count:04d}.png"
        save_path = Path(output_path, class_name, filename)
        save_image(image_with_box, save_path)
        
        image_count += 1
    
    # Show progress every 320 images
    if image_count % 320 == 0:
        print(f"Processed {image_count} images...")

print(f"\nAll done! {image_count} images saved to {output_path}")