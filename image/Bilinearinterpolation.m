img=imread('lena.bmp');

subplot(2,2,1);
imshow(img);%原图
title('source image')
subplot(2,2,2)
downsmpleimg=downsample(img,2);%下采样
imshow(newimg1);
title('downsample')
subplot(2,2,3)
imshow(rescale(downsmpleimg,2));%下采样后插值为2倍
title('2 times')
subplot(2,2,4)
imshow(rescale(downsmpleimg,4));%下采样后插值为4倍
title('4 times')

%下采样函数
function newimg = downsample(img,scale)
    imgsize=size(img);
    newsize =uint16(imgsize/scale)
    newimg = zeros(newsize)
    for i=1:newsize(1)
        for j=1:newsize(2)
            newimg(i,j)=img((i-1)*scale+1,(j-1)*scale+1);
        end
    end
    newimg=uint8(newimg)
end

% 改变大小
function newimg = rescale(img,scale)
    newimg=n(img,scale);
    newsize=size(newimg);
    for i=1:scale:newsize(1)-1
        for j=1:scale:newsize(2)-1
            rect = newimg(i:i+scale,j:j+scale);
            newimg(i:i+scale,j:j+scale)=Bilinearinterpolation(rect);
        end
    end
    newimg=uint8(newimg(1:newsize-1,1:newsize-1));
end


% 插值一张图片
function newimg = n(img,scale)
    oldsize= size(img);
    newscale = oldsize*scale+1;
    newimg = zeros(newscale);
    for i=0:oldsize(1)-1
        for j=0:oldsize(2)-1
            newimg(i*scale+1,j*scale+1)=img(i+1,j+1);
        end
    end
end

% 插值一个方块
function newrect=Bilinearinterpolation(rect)
    rsize = size(rect);
    newrect=rect;
    for i=1:rsize(1)
        for j=1:rsize(2)
            newrect(i,j)=fix(padding(rect,i,j));
        end
    end
end
% 插值一个像素点
function value=padding(rect,x,y)
    rsize = size(rect);
    m1= [rsize(1)-x,x-1];
    m2=m1*[rect(1,1),rect(1,rsize(2));rect(rsize(1),1),rect(rsize(1),rsize(2))];
    m3=m2*[rsize(2)-y;y-1];
    value = m3/( (rsize(1)-1)*(rsize(2)-1));
end

