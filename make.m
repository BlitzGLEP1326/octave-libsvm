function [] = make(clean_up)
  % This make.m is for OCTAVE under Windows, Mac, and Unix
  %
  % This version was derived by Alan Meeson (28/09/2014) 
  % from the make.m for Matlab and Octave by Rong-En Fan and Ting-Fan Wu which was included with LibSVM 3.18
  % See the Readme file in the matlab section of the LibSVM package for further details.

  % Please note: both 32 and 64 bit versions of Octave use the same extension for the resulting oct files.
  mkoctfile svmtrain.cc ../svm.cpp svm_model_octave.cc 
  mkoctfile svmpredict.cc ../svm.cpp svm_model_octave.cc
	
	if exist('clean_up', 'var') && clean_up
		delete('svm.o');
    delete('svmtrain.o');
    delete('svmpredict.o');
    delete('svm_model_octave.o');
	end
end