clear;
close all;
clc;
data=csvread('Dropout_data_final.csv');
data(:,1)=[]; % drop first column
data(1,:)=[]; % drop first row
%data = data1(randperm(size(data1,1)),:);
split=0.6*length(data)
train_X=data(1:split,1:10);
train_y=data(1:split,12);
[m, n] = size(train_X);
train_X = [ones(m, 1) train_X];
test_X=data(split+1:end,1:10);
test_y=data(split+1:end,12);

test_X = [ones((length(test_X)), 1) test_X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);
[cost, grad] = costFunction(initial_theta, train_X, train_y);
fprintf("cost function: %f\n",cost)
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = fminunc(@(t)(costFunction(t, train_X, train_y)), initial_theta, options);
fprintf('Cost at theta found by fminunc: %f\n', cost);
p = predict(theta, test_X);

fprintf('Testing Accuracy: %f\n', mean(double(p == test_y)) * 100);