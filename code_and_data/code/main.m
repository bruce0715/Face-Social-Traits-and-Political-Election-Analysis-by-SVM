%% INSTRUCTIONS:
% Notes: 
% 1). all the root (root is the directory in my computer) variables need to be changed before runing.
% 2). change flags into 1 to train the model or run other time-consuming
% computations and it will save to the root directory. And then you can
% change the flags into 0 to import your result so that you can save your
% time.
% 3).there are two functions written by my own in the second part for using convenience.
% my_feature_prep and myranksvm. The ranksvm function is downloaded online.

%% (SECTION) PREPARATION (ALL the Question divide by (SECTION))
% Mex libsvm functions and Hogfeatures.cc into matlab and set root. 

% prep
clear all;
close all;

% flags
flag_compile_libsvm_c = 1;
flag_compile_libsvm_mex = 1;

% compile libsvm
if flag_compile_libsvm_c
    parent = cd('libsvm-3.21');
    [status,cmdout] = system('make');
    cd(parent);
    disp(status);
    disp(cmdout);
end

if flag_compile_libsvm_mex
    parent = cd('libsvm-3.21/matlab');
    make;
    cd(parent);
end

% setup
diary('P1_1.out');
rng(123);
addpath('libsvm-3.21/matlab');

%bring Hogfetures.cc into matlab
root='/Users/brucezhu/Desktop/UCLA_FA2017/STAT231/project4/project4_code_and_data';
cd(root)
addpath('./libsvm_matlab/');
mex HoGfeatures.cc



%% (SECTION) Q1.1

% data
disp('loading data ...');
load('train-anno.mat', 'face_landmark', 'trait_annotation');
features = face_landmark;
labels = trait_annotation;
clear trait_annotation
clear face_landmark

% clean useless features(have same number) and scale the features
temp_feature=bsxfun(@minus,features,mean(features,1));
features( :, ~any(temp_feature,1) ) = [];
clear temp_feature

for i=1:size(features,2)
    features(:,i)=(features(:,i)-min(features(:,i)))/(max(features(:,i))-min(features(:,i)));
end

disp('down')

%% Train the 14 SVR models with landmark features only for Q1.1

flag.train=1;

if flag.train
    mse=zeros(size(labels,2),1);
    paras2=zeros(size(labels,2),2);

    for t=1:size(labels,2)
    
        %meshgrid search
        [C,epsilon]=meshgrid(-5:2:9,-9:2:-1);
        acc=zeros(numel(C),1);
    
        for i=1:numel(C)
            cmd=['-s 3 -t 0 ',sprintf('-c %f -p %f -v %d',2^C(i),2^epsilon(i),5)];
            acc(i)=libsvmtrain(labels(:,t),features,cmd);
            fprintf('%d traits %d trials',t,i) 
        end
        [~,idx]=min(acc);
        mse(t)=min(acc);
        paras2(t,1)=C(idx);
        paras2(t,2)=epsilon(idx);
    end
    cd(root)
    save('paras2.mat','-v7.3','paras2')
    save('mse.mat','-v7.3','mse')
else
    cd(root)
    load('paras2.mat','paras2')
    load('mse.mat','mse')
end

%% Get the training and testing error using k-fold validation for Q1.1

fold=5; %use 5 fold crossvalidation 
Indices=crossvalind('Kfold',size(labels,1),fold);
mse_train_rec=zeros(fold,size(labels,2));
mse_test_rec=zeros(fold,size(labels,2));

for k=1:fold
    %split train and test features and labels
    train_index=find(Indices~=k);
    test_index=find(Indices==k);
    labels_train=labels(train_index,:);
    features_train=features(train_index,:);
    labels_test=labels(test_index,:);
    features_test=features(test_index,:);
    
    %caculate the training and testing error of specific data split
    for t =1:size(labels,2)
        cmd=['-s 3 -t 0 ',sprintf('-c %f -p %f',2^(paras2(t,1)),2^(paras2(t,2)))];
        model=libsvmtrain(labels_train(:,t),features_train,cmd);
        [~,train_mse,~]=libsvmpredict(labels_train(:,t),features_train,model);
        [~,test_mse,~]=libsvmpredict(labels_test(:,t),features_test,model);
        %record spcific training testing error for one trait
        mse_train_rec(k,t)=train_mse(2);
        mse_test_rec(k,t)=test_mse(2);
    end
end

%caculate the mean train and test error of 14 social traits over 5 fold
%validation.
mse_train=mean(mse_train_rec,1);
mse_test=mean(mse_test_rec,1);

%% Plot train (mse_train) and test (mse_test) error curve for Q1.1

figure();
plot(mse_train,'-o','Color','blue','Linewidth',0.75)
xlabel('Different traits','FontSize',15)
ylabel('Mean squared error','FontSize',15)
hold on 
plot(mse_test,'-o','Color','red','Linewidth',0.75)
legend({'train mse','test mse'},'FontSize',16)
hold off







%% (SECTION) Q1.2 

cd(root)

% get the hog feature of each img.
for i=1:491
    image=double(imread(sprintf('img/M%04d.jpg',i)));
    HogFeat=HoGfeatures(image);
    HogFeat=reshape(HogFeat,size(HogFeat,1)*size(HogFeat,2)*size(HogFeat,3),1);
    HogFeat_rec(i,:)=HogFeat;
end
HogFeat_rec=double(HogFeat_rec);
clear HogFeat

% connect scaled landmark features with Hog features to make a new feature.
features_hog=horzcat(features,HogFeat_rec);

%% Train new 14 SVR models with landmark+hog features for Q1.2

flag.train_hog=1;

if flag.train_hog
    mse_hog=zeros(size(labels,2),1);
    paras2_hog=zeros(size(labels,2),2);
    for t=1:size(labels,2)
        [C_hog,epsilon_hog]=meshgrid(-15:2:5,-9:2:5);
        acc_hog=zeros(numel(C_hog),1);
    
        for i=1:numel(C_hog)
            cmd_hog=['-s 3 -t 0 ',sprintf('-c %f -p %f -v %d',2^C_hog(i),2^epsilon_hog(i),5)];
            acc_hog(i)=libsvmtrain(labels(:,t),features_hog,cmd_hog);
            fprintf('%d traits %d trials',t,i) 
        end
        [~,idx_hog]=min(acc_hog);
        mse_hog(t)=min(acc_hog);
        paras2_hog(t,1)=C_hog(idx_hog);
        paras2_hog(t,2)=epsilon_hog(idx_hog);
    end
    cd(root)
    save('paras2_hog2.mat','-v7.3','paras2_hog')
    save('mse_hog2.mat','-v7.3','mse_hog')
else
    cd(root)
    load('paras2_hog2.mat','paras2_hog')
    load('mse_hog2.mat','mse_hog')
end

%% Get the training and testing error using k-fold validation for Q1.2

%set flag.hog_kvalid=1 if you need caclulate, otherwise just load from file
flag.hog_kvalid=1;

if flag.hog_kvalid
    fold_hog=5;
    Indices_hog=crossvalind('Kfold',size(labels,1),fold_hog);
    mse_train_rec_hog=zeros(fold_hog,size(labels,2));
    mse_test_rec_hog=zeros(fold_hog,size(labels,2));

    for k=1:fold_hog
        train_index_hog=find(Indices_hog~=k);
        test_index_hog=find(Indices_hog==k);
        labels_train_hog=labels(train_index_hog,:);
        features_train_hog=features_hog(train_index_hog,:);
        labels_test_hog=labels(test_index_hog,:);
        features_test_hog=features_hog(test_index_hog,:);

        for t =1:size(labels,2)
            cmd_hog=['-s 3 -t 0 ',sprintf('-c %f -p %f',2^(paras2_hog(t,1)),2^(paras2_hog(t,2)))];
            model_hog=libsvmtrain(labels_train_hog(:,t),features_train_hog,cmd_hog);
            [~,train_mse_hog,~]=libsvmpredict(labels_train_hog(:,t),features_train_hog,model_hog);
            [~,test_mse_hog,~]=libsvmpredict(labels_test_hog(:,t),features_test_hog,model_hog);
            mse_train_rec_hog(k,t)=train_mse_hog(2);
            mse_test_rec_hog(k,t)=test_mse_hog(2);
        end

    end
    mse_train_hog=mean(mse_train_rec_hog,1);
    mse_test_hog=mean(mse_test_rec_hog,1);
    save('mse_train_hog.mat','-v7.3','mse_train_hog')
    save('mse_test_hog.mat','-v7.3','mse_test_hog')
else
    load('mse_train_hog.mat','mse_train_hog')
    load('mse_test_hog.mat','mse_test_hog')
end

%% Plot train and test error curve for Q1.2
figure();
plot(mse_train_hog,'-o','Color','blue','Linewidth',0.75)
xlabel('Different traits','FontSize',15)
ylabel('Mean squared error','FontSize',15)
hold on 
plot(mse_test_hog,'-o','Color','red','Linewidth',0.75)
hold on
plot(mse_train,'b--*','Color','blue','Linewidth',0.75)
hold on
plot(mse_test,'b--*','Color','red','Linewidth',0.75)
legend({'train mse rich','test mse rich','train mse poor','test mse poor'},'FontSize',16)
hold off





%% (SECTION) Q2.1

%set root directory
root='/Users/brucezhu/Desktop/UCLA_FA2017/STAT231/project4/project4_code_and_data';
cd(root)

%get Hogfeatures function
addpath('./libsvm_matlab/');
mex HoGfeatures.cc

%% Governors for Q2.1

%set train.gov=1 to run set 0 to skip
train.gov=1;

if train.gov
    % use my own function 'my_feature_prep' to preprocess data for ranksvm.
    [gov_features,gov_labels]=my_feature_prep('gov',root);
    
    % set the row numbers index in advance for 4 fold validation.
    gov_index=[1,29,57,85,113];
    
    % use own function 'myranksvm' to train ranksvm model with 4 fold
    % validation. And return average training and testing accuracy and best
    % parameter C to achieve this result
    [accuracy_gov,c_choice_gov]=myranksvm(gov_features,gov_labels,gov_index);
end

%% Senators for Q2.1

%set train.sen=1 to run set 0 to skip
train.sen=1;

if train.sen
    % use my own function 'my_feature_prep' to preprocess data for ranksvm.
    [sen_features,sen_labels]=my_feature_prep('sen',root);
    
    % set the row numbers index in advance for 4 fold validation.
    sen_index=[1,29,57,85,117];
    
    % use own function 'myranksvm' to train ranksvm model with 4 fold
    % validation. And return average training and testing accuracy and best
    % parameter C to achieve this result
    [accuracy_sen,c_choice_sen]=myranksvm(sen_features,sen_labels,sen_index);
end





%% (SECTION) Q2.2 

% set root directory
root='/Users/brucezhu/Desktop/UCLA_FA2017/STAT231/project4/project4_code_and_data';

% set flag.extractfeatures=1 to use model in Q1.2 to predict the 14 social
% traits for governors and senators.
flag.extractfeatures=1;

if flag.extractfeatures
    cd(root)
    % get the best parameters of the 14 SVR saved by Q1.2
    load('paras2_hog2.mat','paras2_hog')
    gov_traits_rec=zeros(size(gov_features,1),14);
    sen_traits_rec=zeros(size(sen_features,1),14);
    
    % predict 14 social traits in order.
    for t =1:14
        cmd=['-s 3 -t 0 ',sprintf('-c %f -p %f',2^(paras2_hog(t,1)),2^(paras2_hog(t,2)))];
        
        % train model using 'labels' and 'features_hog' get by Q1.2
        model=libsvmtrain(labels(:,t),features_hog,cmd);
        
        % predict social traits for gov and sen using 'gov_features' 
        % and 'sen_features' get by Q2.1
        gov_traits=libsvmpredict(ones(size(gov_features,1),1),gov_features,model);
        sen_traits=libsvmpredict(ones(size(sen_features,1),1),sen_features,model);
        
        % record the traits_features for each t
        gov_traits_rec(:,t)=gov_traits;
        sen_traits_rec(:,t)=sen_traits;
    end
    cd(root)
    save('gov_traits_rec.mat','-v7.3','gov_traits_rec')
    save('sen_traits_rec.mat','-v7.3','sen_traits_rec')
else
    cd(root)
    load('gov_traits_rec.mat','gov_traits_rec')
    load('sen_traits_rec.mat','sen_traits_rec')
end

%% Governors for Q2.2 using 14 predicted social traits

% set the row numbers index in advance for 4 fold validation.
gov_index=[1,29,57,85,113];

% get the 'gov_labels' same as Q2.1 for predicting governors election result
[~,gov_labels]=my_feature_prep('gov',root);

% use 14 predicted social traits to predict election result for governors.
[gov_traits_acc,gov_traits_c]=myranksvm(gov_traits_rec,gov_labels,gov_index);

%% Senators for Q2.2 using 14 predicted social traits

%set the row numbers index in advance for 4 fold validation.
sen_index=[1,29,57,85,117];

% get the 'sen_labels' same as Q2.1 for predicting senator election result
[~,sen_labels]=my_feature_prep('sen',root);

% use 14 predicted social traits to predict election result for senators.
[sen_traits_acc,sen_traits_c]=myranksvm(sen_traits_rec,sen_labels,sen_index);







%% (SECTION) Q2.3
% calculate the correlation between each face attribute (F_win-F_lose) and
% vote difference

%For governor
win_index_gov=2:2:size(gov_traits_rec,1);
lose_index_gov=1:2:size(gov_traits_rec,1);
gov_trait_diff=gov_traits_rec(win_index_gov,:)-gov_traits_rec(lose_index_gov,:);

load('stat-gov.mat','vote_diff')
vote_diff2=vote_diff(win_index_gov,:);

%caculation correlation for gov 
cor_result_rec_gov=zeros(size(gov_trait_diff,2),1);
for i=1:size(gov_trait_diff,2)
cor_result=corrcoef(gov_trait_diff(:,i),vote_diff2);
cor_result_rec_gov(i)=cor_result(2,1);
end

%For senator
win_index_sen=2:2:size(sen_traits_rec,1);
lose_index_sen=1:2:size(sen_traits_rec,1);
sen_trait_diff=sen_traits_rec(win_index_sen,:)-sen_traits_rec(lose_index_sen,:);

load('stat-sen.mat','vote_diff')
vote_diff3=vote_diff(win_index_sen,:);

%caculation correlation for senator
cor_result_rec_sen=zeros(size(sen_trait_diff,2),1);
for i=1:size(sen_trait_diff,2)
cor_result=corrcoef(sen_trait_diff(:,i),vote_diff3);
cor_result_rec_sen(i)=cor_result(2,1);
end

%draw the radar plot using Excel.
