function [features,labels]=my_feature_prep(index,root)
cd(root)
if index=='gov'
    x=load('stat-gov.mat');
end

if index=='sen'
    x=load('stat-sen.mat');
end

%get features,we treat FA-FB,FB-FA as features
landmark=x.face_landmark;
index_loss=1:2:size(landmark,1);
index_win=2:2:size(landmark,1);
features=zeros(size(landmark));
features(index_loss,:)=landmark(index_loss,:)-landmark(index_win,:);
features(index_win,:)=landmark(index_win,:)-landmark(index_loss,:);

%get hog features of gov
cd (root)

%get hog features of gov
for i=1:size(landmark,1)
    if index=='gov'
        image=double(imread(sprintf('img-elec/governor/G%04d.jpg',i)));
    else
        image=double(imread(sprintf('img-elec/senator/S%04d.jpg',i)));
    end
    HogFeat=HoGfeatures(image);
    HogFeat=reshape(HogFeat,size(HogFeat,1)*size(HogFeat,2)*size(HogFeat,3),1);
    HogFeat_rec(i,:)=HogFeat;
end

HogFeat_rec=double(HogFeat_rec);

%del same columns and scale the features
del=find(max(features,[],1)-min(features,[],1)==0);
features(:,del)=[];

newfeatures=bsxfun(@minus,features,min(features,[],1));
newfeatures=bsxfun(@times,newfeatures,1./(max(features,[],1)-min(features,[],1)));
features=newfeatures;

%connect features with Hog features
features=horzcat(features,HogFeat_rec);

%construct label matrix
labels=zeros(size(x.vote_diff,1)/2,size(x.vote_diff,1));

for i=1:(size(x.vote_diff,1)/2)
    labels(i,2*i-1)=sign(x.vote_diff(2*i-1));
    labels(i,2*i)=sign(x.vote_diff(2*i));
end