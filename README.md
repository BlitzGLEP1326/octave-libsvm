# octave-libsvm : Octave interface for LIBSVM

This tool provides an octave specific interface for LIBSVM, a library 
for support vector machines (http://www.csie.ntu.edu.tw/~cjlin/libsvm). 
The tool was derived directly from the Matlab/Octave interface provided 
in the main LibSVM package.

This tool differs from the Matlab/Octave version in that it uses the
Octave API functions rather than the Matlab API functions.  This 
permits use with MXE 64 bit builds of Octave which rely on mingw32.

Initial testing also shows a minor speed increase in comparison to the
Matlab/Octave version, bringing its performance in line with that of 
the Matlab/Octave version as executed by Matlab.

Currently the functions have been tested for binary linear SVM only, and 
show identical results to those from the Matlab/Octave functions.

# Installation

The code provided here works in conjunction with the existing LIBSVM
library, which is available at http://www.csie.ntu.edu.tw/~cjlin/libsvm

Pre-built binary files are not provided, so compilation is required.
To compile, download the LIBSVM library, then place this code in a 
subfolder named 'octave', this will be in the same place as a folder
named 'matlab'.  From here compilation can be performed using the make.m 
script. The make.m script, if compilation is successful, produces 
svmtrain.oct and svmpredict.oct.

For further information on compiling oct files, please refer to:v
https://www.gnu.org/software/octave/doc/interpreter/Getting-Started-with-Oct_002dFiles.html

# Usage, Model Structure and Prediction Results

The oct functions produced are designed to be compatible in performance 
with those provided by the Matlab/Octave interface. Please refer to the
documentation for the Matlab/Octave interface.

Currently the functions have been tested for binary linear SVM only, and 
show identical results to those from the Matlab/Octave functions.

If you intend to use other kernels, I would suggest testing first and
comparing to the results produced using the matlab interface if possible.

# Additional Information

This Octave interface was created by Alan Meeson based on the existing
Matlab/Octave interface.

The Matlab/Octave interface was initially written by Jun-Cheng Chen, 
Kuan-Jen Peng, Chih-Yuan Yang and Chih-Huai Cheng from Department of 
Computer Science, National Taiwan University. The current version was 
prepared by Rong-En Fan and Ting-Fan Wu. 

If you find this tool useful, please cite LIBSVM as follows:

Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for support
vector machines. ACM Transactions on Intelligent Systems and
Technology, 2:27:1--27:27, 2011. Software available at
http://www.csie.ntu.edu.tw/~cjlin/libsvm

For any question regarding LIBSVM, please contact Chih-Jen Lin 
<cjlin@csie.ntu.edu.tw>, or check the FAQ page:

http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#/Q9:_MATLAB_interface

For questions specific to this Octave interface, please contact 
Alan Meeson <alan@carefullycalculated.co.uk>