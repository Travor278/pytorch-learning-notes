import os

root_dir = 'dataset/hymenoptera_data/train'
target_classes = ['ants', 'bees']

for class_name in target_classes:
    image_dir = class_name + '_image'
    label_dir = class_name + '_label'
    
    full_image_dir = os.path.join(root_dir, image_dir)
    img_list = os.listdir(full_image_dir)
    
    for img in img_list:
        file_name = img.split('.jpg')[0]
        
        out_path = os.path.join(root_dir, label_dir, "{}.txt".format(file_name))
        
        with open(out_path, 'w') as f:
            f.write(class_name)
