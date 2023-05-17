clear all, close all
ModelImage   = 'SGM_0767.jpg';

Num_Masks=2;
 
X_original   = imread(ModelImage);
figure,imshow(X_original);


X = double(X_original);
sX1=size(X_original,1);
sX2=size(X_original,2);


X = reshape(X,size(X,1)*size(X,2),3);
M = mean(X);
X = phi_mcs(X,M,[]);
[t,P,r2,ma,s]=phi_kpcaV(X,2,0);
figure,
plot(t(:,1),t(:,2),'.')

%%

MinT = min(t);
MaxT = max(t);
t(:,1)=round((t(:,1)-MinT(1))./(MaxT(1)-MinT(1))*255);
t(:,2)=round((t(:,2)-MinT(2))./(MaxT(2)-MinT(2))*255);


 MASKS={};
 T2MASK{1}=t;
 T2MASK{2}=t;
 T2MASK{3}=t;
 T2MASK{4}=t;
 MaskNames={'Mask1';'Mask2';'Mask3';'Mask4'};
 MaskColors='rgbw';
 for m=1:Num_Masks
     
     [B,EdX,EdY]=phi_hist2d(T2MASK{m}(:,1),[0 255],256/2,T2MASK{m}(:,2),[0 255],256/2,1);hold on
     title(['Please choose mask for ',MaskNames{m}])
     button=1;
     vx=[];vy=[];
     while button==1
         [x,y,button]=ginput(1);
         if button==1
             plot(x,y,'oy','MarkerFaceColor','y')
             vx=[vx;x];vy=[vy;y];
         end
     end
     if ~isempty(vx)
         for i=1:length(vx)-1, plot([vx(i); vx(i+1)],[vy(i) ;vy(i+1)],'-w'),end
         plot([vx(end); vx(1)],[vy(end) ;vy(1)],'-w')
     end
     bw = poly2mask(vx, vy, 256/2, 256/2);
     mask.mask=bw;
     mask.vertices.x=vx;
     mask.vertices.y=vy;
     MASKS{end+1,1}=mask;
 end
 drawnow
 
 [B]=phi_hist2d(t(:,1),[0 255],256/2,t(:,2),[0 255],256/2,1);hold on
  
 for m=1:Num_Masks
     vx=MASKS{m}.vertices.x;
     vy=MASKS{m}.vertices.y;
 for i=1:length(vx)-1, plot([vx(i); vx(i+1)],[vy(i) ;vy(i+1)],['o-',MaskColors(m)],'MarkerFaceColor',MaskColors(m)),end
         plot([vx(end); vx(1)],[vy(end) ;vy(1)],['o-',MaskColors(m)],'MarkerFaceColor',MaskColors(m))
 end
 
 [Bindx,B2] = phi_hist2d_membership(t,EdX,EdY);
 
mask4image=zeros(size(t,1),1);

    
 for m=1:Num_Masks
     aux=Bindx(MASKS{m}.mask);
     aux=cell2mat(aux');
     aux=aux';
     mask4image(aux)=m;
 end

XMask=mask4image;
XMask=uint8(reshape(XMask,sX1,sX2,1));
figure, imagesc(XMask)