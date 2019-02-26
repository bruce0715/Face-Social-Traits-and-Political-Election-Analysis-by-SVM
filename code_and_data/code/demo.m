addpath('./libsvm_matlab/');
mex HoGfeatures.cc

im = imread('img/M0005.jpg');
im = double(im);
hogfeat = HoGfeatures(im);
imshow(drawHOGtemplates(hogfeat, [61 61]));

