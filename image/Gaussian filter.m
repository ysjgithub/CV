img=[];
a=1:256;
b=256:-1:1;

for i=1:256
    if i<=64
        img=[img;a];
    end
    if (64<i)&&(i<=128)
        img=[img;b];
    end
    if (128<i)&&(i<=192)
        img=[img;a];
    end
    if (192<i)&&(i<=256)
        img=[img;b];
    end
end
img=img/256
subplot(1,2,1)
imshow(img)
title('source map')

size =5 %核尺寸
sigma=1 %标准差
core=getcore(sigma,size)%根据sigma和size得到卷积核
newimg = zeros(256+size,256+size)%补0
newimg(3:258,3:258)=img;
gaos=zeros(256,256);
for i=3:258
    for j=3:258
        mode = newimg(i-2:i+2,j-2:j+2);
        res=conv(mode,core);
        gaos(i-2,j-2)=res;
    end
end
subplot(1,2,2);
imshow(gaos)
title('smooth map')

% 根据sigma和size得到卷积核
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
% 卷积操作
function res= conv(mode,core)
    res=sum(sum(mode.*core));
end








