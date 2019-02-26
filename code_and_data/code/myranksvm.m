function [max_acc,c_best]=myranksvm(features,labels,index)

%fold number
fold=4;
n=size(labels,2);
p=size(labels,1);
% parameter range
C=-10:10;
acc_rec=zeros(numel(C),2);

for c=1:numel(C)
    acc_rec1=zeros(fold,2);
    for k=1:fold
        % split dataset into training and testing
        features_train=features;
        features_train(index(k):(index(k+1)-1),:)=[];
        features_test=features(index(k):(index(k+1)-1),:);
    
        labels_train=labels;
        labels_train((index(k)+1)/2:(index(k+1)-1)/2,:)=[];
        labels_train(:,index(k):(index(k+1)-1))=[];
        
        % perform ranksvm to get w
        w=ranksvm(features_train,labels_train,ones(size(labels_train,1),1)*(2^C(c)));
        
        result_train=features_train*w;
        
        % use winner minus loser on training data to see if the value bigger than 0? 
        cmp_train=result_train(2:2:size(result_train,1))-result_train(1:2:size(result_train,1));
        acc_train=size(cmp_train(cmp_train>0),1)/size(cmp_train,1);
        
        % use winner minus loser on testing data to see if the value bigger than 0? 
        result_test=features_test*w;
        cmp_test=result_test(2:2:size(result_test,1))-result_test(1:2:size(result_test,1));
        acc_test=size(cmp_test(cmp_test>0),1)/size(cmp_test,1);
        
        %record accuracy
        acc_rec1(k,1)=acc_train;
        acc_rec1(k,2)=acc_test;
    end
    % average the training and testing accuracy over 4 fold
    acc_rec(c,:)=mean(acc_rec1,1);
end 
% select biggest test_accuracy, if test_accuracy are same, then consider training accuracy
[~,idx_sen]=max(acc_rec(:,2)+0.0000001*acc_rec(:,1)); 
% return max accuracy
max_acc=acc_rec(idx_sen,:);
% return best parameters
c_best=C(idx_sen);
