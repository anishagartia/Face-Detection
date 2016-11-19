img_path = '../data/caltech_faces/Caltech_CropFaces/caltech_web_crop_00001.jpg';
img = im2single(imread(img_path));
commandStr = 'python test_model.py img_path  1 1 36 36';
res = system(commandStr);
