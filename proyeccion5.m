%%
figure
I = imread('image2cal.jpg');
imshow(I)
impixelinfo
size(I)
%%
clear
clc
I = imread('image2cal.jpg');
figure
imshow(I)
impixelinfo
hold on
cero = zeros(1,4);

p = [260,328;
    160,239;
    260,229;
    394,323;
    129,331;
    379,224];
P = [1,1,0,1;
    1.5,3,0,1;
    2.5,2.5,0,1;
    1.5,0.5,0,1;
    0.5,1.5,0,1;
    3.5,1.5,0,1];
n = size(P,1);
W = [];
for i=1:n
    W = [W;
        P(i,:) cero -p(i,1)*P(i,:);
        cero P(i,:) -p(i,2)*P(i,:)];
end
Wt = transpose(W);
A = Wt*W;
[V,D] = eig(A);
d = eig(A);
val_opt = find(min(d));
vec_opt = V(:,val_opt);
m1 = vec_opt(1:4);
m2 = vec_opt(5:8);
m3 = vec_opt(9:12);
u = zeros(1,n);
v = zeros(1,n);
for i=1:n
    u(i)=dot(m1,P(i,:))/dot(m3,P(i,:));
    v(i)=dot(m2,P(i,:))/dot(m3,P(i,:));
end
%figure
for i = 1:6
    plot(p(i,1),p(i,2),'.', 'MarkerSize', 30, 'LineWidth', 2,'color','red');
    hold on
end
%% MALLA ARTIFICIAL
%-----------------------piso-----------------------
figure
for g = 0:0.1:5
for i = 0:0.1:5

   plot3([g g], [0 5], [0, 0],'color','red')
   hold on
end
end
for g = 0:0.1:5
for i = 0:0.1:5

   plot3([0 5], [g g], [0, 0],'color','red')
   hold on
end
end
xlabel('eje x'); ylabel('eje x'); zlabel('eje z');
%%
paso = 0.1;
total = round(5/paso)
x11 = linspace(0,5,100);
zx11 = zeros(1,100); 
counter = 0;
for i = 0:total
    zx11 = [zx11;
        counter*ones(1,100)];
    counter = counter + paso;
end
figure
for i = 1:total+2
    plot(x11,zx11(i,:),'color','red')
    hold on
    axis equal
end

x22 = linspace(0,5,100);
zx22 = zeros(1,100); 

counter = 0;
for i = 0:total
    zx22 = [zx22;
        counter*ones(1,100)];
    counter = counter + paso;
    
end

for i = 1:total+2
    plot(zx22(i,:),x22,'color','red')
    hold on
    axis equal
end
%% COORDENADAS CON 4 POSICIONES
%-----------------------------------------------------
zx_3d11 = [];
for i = 1:total
for j = 1:100
zx_3d11 = [zx_3d11;
     x11(j) zx11(i,j) 0 1];
end
end

zx_3d22 = [];
for i = 1:total
for j = 1:100
zx_3d22 = [zx_3d22;
    zx22(i,j) x22(j) 0 1];
end
end
%---------------------------------------------------------

%% GENERACIÓN DE LAS COORDENADAS U Y V
%-----------------------------------------------------
u22 = zeros(1,5000);
v22 = zeros(1,5000);
for i=1:5000
    u22(i)=dot(m1,zx_3d11(i,:))/dot(m3,zx_3d11(i,:));
    v22(i)=dot(m2,zx_3d11(i,:))/dot(m3,zx_3d11(i,:));
end
matriz_with_info = zeros(5000,4);
matriz_with_info(:,1)=u22;
matriz_with_info(:,2)=v22;
matriz_with_info(:,3)= zx_3d11(:,1);
matriz_with_info(:,4)= zx_3d11(:,2);

u33 = zeros(1,5000);
v33 = zeros(1,5000);
for i=1:5000
    u33(i)=dot(m1,zx_3d22(i,:))/dot(m3,zx_3d22(i,:));
    v33(i)=dot(m2,zx_3d22(i,:))/dot(m3,zx_3d22(i,:));
end
matriz_with_info_2 = zeros(5000,4);
matriz_with_info_2(:,1)=u33;
matriz_with_info_2(:,2)=v33;
matriz_with_info_2(:,3)= zx_3d22(:,1);
matriz_with_info_2(:,4)= zx_3d22(:,2);

format long g
matrix_final = [matriz_with_info;matriz_with_info_2];
dlmwrite('matrix3.txt',matrix_final)
%-----------------------------------------------------------------


%% SOBRELAPAMIENTO

figure
I = imread('image2cal.jpg');
imshow(I);
impixelinfo
axis on
hold on;
%-------------------------------------------------------------------------------
for i = 1:5000
    plot(u22(i),v22(i), '.', 'MarkerSize', 5, 'LineWidth', 2,'color','red');
    hold on
end

for i = 1:5000
    plot(u33(i),v33(i), '.', 'MarkerSize', 5, 'LineWidth', 2,'color','red');
    hold on
end
%%
figure
I = imread('image2cal.jpg');
imshow(I);
axis on
hold on;
%-------------------------------------------------------------------------------
for i = 1:700
    plot(u22(i),v22(i), '.', 'MarkerSize', 5, 'LineWidth', 2,'color','red');
    hold on
end

for i = 1:700
    plot(u33(i),v33(i), '.', 'MarkerSize', 5, 'LineWidth', 2,'color','red');
    hold on
end