clear;
close all;
clc;
data=csvread('~/Dropout_data_final.csv');
data(:,1)=[]; % drop first column
data(1,:)=[]; % drop first row
%data = data1(randperm(size(data1,1)),:);
split=0.6*length(data);
train_X=data(1:split,1:10);
train_y=data(1:split,12);
[m, n] = size(train_X);
%train_X = [ones(m, 1) train_X];
test_X=data(split+1:end,1:10);
test_y=data(split+1:end,12);

%est_X = [ones((length(test_X)), 1) test_X];
input_layer_size  = 10;  % 20x20 Input Images of Digits
hidden_layer_size = 5;   % 25 hidden units
num_labels = 1;          % 10 labels, from 1 to 10   
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
lambda = 1;
checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = nnCostFunction(initial_nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, train_X, train_y, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' ...
         '\n(for lambda = 3, this value should be about 0.576051)\n\n'], lambda, debug_J);

fprintf('Program paused. Press enter to continue.\n');
pause;
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  You should also try different values of lambda
lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, train_X, train_y, lambda);
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
pred = predict_student_attrition(Theta1, Theta2, test_X);

fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred == test_y)) * 100);