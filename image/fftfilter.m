img = imread('stripe.png')
imgsize=size(img)
row=imgsize(1);
col=imgsize(2);
special_filter_img=getcore(1,5);
h=fspecial('gaussian',[5,5],1);
H=freqz2(h,row,col);
H=fftshift(H);
F=fft2(img,row,col);
G=H.*F;
g=real(ifft2(G));
newimg=uint8(g(1:row,2:col));
subplot(1,2,1);
imshow(newimg);
subplot(1,2,2);
imshow(img);
function core=getcore(sigma,size)
    core=zeros(size,size)
    sumc=0
    for i=1:5
        for j=1:5
            core(i,j)=exp(((i-3)^2+(j-3)^2)/-(2*sigma^2))/(2*pi*sigma^2)
            sumc=sumc+core(i,j)
        end
    end
    core=core/sumc
end