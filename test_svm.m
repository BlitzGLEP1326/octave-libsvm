function [times] = test_svm()
%test_svm Performs a nonsense analysis of random data to test the compiled code.
%This can be used to compare version of the LIBSVM library.
% times = test_svm ()
%	times - the time taken to perform 100 train/test steps.

%Created by Alan Meeson 2nd October 2014

t=tic();
for j = 1:100
  data = [rand(2000, 150); (rand(2000, 150) + 0.5)];
  label = [(zeros(2000, 1) -1) ; ones(2000,1)];
  command = sprintf('-s 0 -c %f -t %u -w1 %f -w-1 1 -e %f -q', 1, 0, 1, 0.001); 

  model = svmtrain(label, data, command);
  [predicted_labels, accuracy, probabilities] = svmpredict(label, data, model, '-q');
end
times = toc(t);
end