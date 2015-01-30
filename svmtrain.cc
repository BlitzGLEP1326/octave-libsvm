/**
 * Octave LIBSVM wrapper
 * This file contains the wrapper function for training an SVM model.
 * 
 * Adapted by Alan Meeson 2nd October 2014 based on the matlab version included with the LIBSVM library.
 * Last Updated by Alan Meeson 7th December 2014
 */

#include <octave/oct.h>
#include <octave/dMatrix.h>
#include <octave/dColVector.h>
#include <octave/dRowVector.h>
#include <octave/ov-struct.h>
#include <string.h>
#include "../svm.h"

#include "svm_model_octave.h"

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_null(const char *s) {}
//void print_string_octave(const char *s) {printf(s);}

void exit_with_help()
{
	printf(
	"Usage: model = svmtrain(training_label_vector, training_instance_matrix, 'libsvm_options');\n"
	"libsvm_options:\n"
	"-s svm_type : set type of SVM (default 0)\n"
	"	0 -- C-SVC		(multi-class classification)\n"
	"	1 -- nu-SVC		(multi-class classification)\n"
	"	2 -- one-class SVM\n"
	"	3 -- epsilon-SVR	(regression)\n"
	"	4 -- nu-SVR		(regression)\n"
	"-t kernel_type : set type of kernel function (default 2)\n"
	"	0 -- linear: u'*v\n"
	"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
	"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
	"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
	"	4 -- precomputed kernel (kernel values in training_instance_matrix)\n"
	"-d degree : set degree in kernel function (default 3)\n"
	"-g gamma : set gamma in kernel function (default 1/num_features)\n"
	"-r coef0 : set coef0 in kernel function (default 0)\n"
	"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
	"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-m cachesize : set cache memory size in MB (default 100)\n"
	"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
	"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
	"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
	"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
	"-v n : n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	);
}

// svm arguments
struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
struct svm_node *x_space;
int cross_validation;
int nr_fold;


double do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double,prob.l);
	double retval = 0.0;

	svm_cross_validation(&prob,&param,nr_fold,target);
	if(param.svm_type == EPSILON_SVR ||
	   param.svm_type == NU_SVR)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			);
		retval = total_error/prob.l;
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
		retval = 100.0*total_correct/prob.l;
	}
	free(target);
	return retval;
}

// nrhs should be 3
int parse_command_line(int nrhs, const octave_value_list prhs, char *model_file_name)
{
	int i, argc = 1;
	char cmd[CMD_LEN];
	char *argv[CMD_LEN/2];
	void (*print_func)(const char *) = print_null;//print_string_octave;	// default printing to matlab display

	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	cross_validation = 0;

	if(nrhs <= 1)
		return 1;

	if(nrhs > 2)
	{
		// put options in argv[]
		strncpy(cmd, prhs(2).string_value().c_str(), CMD_LEN);
		//mxGetString(prhs[2], cmd, mxGetN(prhs[2]) + 1);
		if((argv[argc] = strtok(cmd, " ")) != NULL)
			while((argv[++argc] = strtok(NULL, " ")) != NULL)
				;
	}

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		++i;
		if(i>=argc && argv[i-1][1] != 'q')	// since option -q has no parameter
			return 1;
		switch(argv[i-1][1])
		{
			case 's':
				param.svm_type = atoi(argv[i]);
				break;
			case 't':
				param.kernel_type = atoi(argv[i]);
				break;
			case 'd':
				param.degree = atoi(argv[i]);
				break;
			case 'g':
				param.gamma = atof(argv[i]);
				break;
			case 'r':
				param.coef0 = atof(argv[i]);
				break;
			case 'n':
				param.nu = atof(argv[i]);
				break;
			case 'm':
				param.cache_size = atof(argv[i]);
				break;
			case 'c':
				param.C = atof(argv[i]);
				break;
			case 'e':
				param.eps = atof(argv[i]);
				break;
			case 'p':
				param.p = atof(argv[i]);
				break;
			case 'h':
				param.shrinking = atoi(argv[i]);
				break;
			case 'b':
				param.probability = atoi(argv[i]);
				break;
			case 'q':
				print_func = &print_null;
				i--;
				break;
			case 'v':
				cross_validation = 1;
				nr_fold = atoi(argv[i]);
				if(nr_fold < 2)
				{
					printf("n-fold cross validation: n must >= 2\n");
					return 1;
				}
				break;
			case 'w':
				++param.nr_weight;
				param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
				param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
				param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
				param.weight[param.nr_weight-1] = atof(argv[i]);
				break;
			default:
				printf("Unknown option -%c\n", argv[i-1][1]);
				return 1;
		}
	}

	svm_set_print_string_function(print_func);

	return 0;
}

// read in a problem (in svmlight format)
int read_problem_dense(ColumnVector &label_vec, Matrix &instance_mat)
{
	int i, j, k;
	int elements, max_index, sc, label_vector_row_num;
	double *samples, *labels;

	prob.x = NULL;
	prob.y = NULL;
	x_space = NULL;

	labels = (double*)label_vec.mex_get_data();//mxGetPr(label_vec);
	samples = (double*)instance_mat.mex_get_data();
	sc = (int)instance_mat.cols();

	elements = 0;
	// the number of instance
	prob.l = (int)instance_mat.rows();
	label_vector_row_num = (int)label_vec.rows();

	if(label_vector_row_num!=prob.l)
	{
		printf("Length of label vector does not match # of instances.\n");
		return -1;
	}

	if(param.kernel_type == PRECOMPUTED)
		elements = prob.l * (sc + 1);
	else
	{
		for(i = 0; i < prob.l; i++)
		{
			for(k = 0; k < sc; k++)
				if(samples[k * prob.l + i] != 0)
					elements++;
			// count the '-1' element
			elements++;
		}
	}

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node, elements);

	max_index = sc;
	j = 0;
	for(i = 0; i < prob.l; i++)
	{
		prob.x[i] = &x_space[j];
		prob.y[i] = labels[i];

		for(k = 0; k < sc; k++)
		{
			if(param.kernel_type == PRECOMPUTED || samples[k * prob.l + i] != 0)
			{
				x_space[j].index = k + 1;
				x_space[j].value = samples[k * prob.l + i];
				j++;
			}
		}
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				printf("Wrong input format: sample_serial_number out of range\n");
				return -1;
			}
		}

	return 0;
}

int read_problem_sparse(ColumnVector &label_vec, SparseMatrix &instance_mat)
{
	int i, j, k, low, high;
	octave_idx_type *ir, *jc;
	int elements, max_index, num_samples, label_vector_row_num;
	double *samples, *labels;
	SparseMatrix instance_mat_col = instance_mat.transpose(); // transposed instance sparse matrix

	prob.x = NULL;
	prob.y = NULL;
	x_space = NULL;

	// each column is one instance
	labels = (double*)label_vec.mex_get_data();
	samples = (double*)instance_mat_col.mex_get_data();
	ir = instance_mat_col.mex_get_ir();
	jc = instance_mat_col.mex_get_jc();

	num_samples = (int)instance_mat_col.nzmax();

	// the number of instance
	prob.l = (int)instance_mat_col.cols();
	label_vector_row_num = (int)label_vec.rows();

	if(label_vector_row_num!=prob.l)
	{
		printf("Length of label vector does not match # of instances.\n");
		return -1;
	}

	elements = num_samples + prob.l;
	max_index = (int)instance_mat_col.rows();

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node, elements);

	j = 0;
	for(i=0;i<prob.l;i++)
	{
		prob.x[i] = &x_space[j];
		prob.y[i] = labels[i];
		low = (int)jc[i], high = (int)jc[i+1];
		for(k=low;k<high;k++)
		{
			x_space[j].index = (int)ir[k] + 1;
			x_space[j].value = samples[k];
			j++;
	 	}
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	return 0;
}

static void fake_answer(int nlhs, octave_value_list &plhs)
{
	int i;
	for(i=0;i<nlhs;i++) plhs(i) = Matrix(0,0);
}

// Interface function of matlab
// now assume prhs[0]: label prhs[1]: features
//void mexFunction( int nlhs, mxArray *plhs[],
//		int nrhs, const mxArray *prhs[] )
//{
DEFUN_DLD (svmtrain, prhs, nargout,
           "[model] = svmtrain(training_labels, training_data, command_line)")
{
	const char *error_msg;
	octave_value_list plhs(nargout);
	// fix random seed to have same results for each run
	// (for cross validation and probability estimation)
	srand(1);
	int nlhs = nargout;
	int nrhs = prhs.length();
	if(nlhs > 1)
	{
		exit_with_help();
		fake_answer(nlhs, plhs);
		return plhs;
	}

	// Transform the input Matrix to libsvm format
	if(nrhs > 1 && nrhs < 4)
	{
		int err;

		if(!prhs(0).is_double_type() || !prhs(1).is_double_type()) {
			printf("Error: label vector and instance matrix must be double\n");
			fake_answer(nlhs, plhs);
			return plhs;
		}

		if(parse_command_line(nrhs, prhs, NULL))
		{
			exit_with_help();
			svm_destroy_param(&param);
			fake_answer(nlhs, plhs);
			return plhs;
		}

		if(prhs(1).is_sparse_type())
		{
			if(param.kernel_type == PRECOMPUTED)
			{
				// precomputed kernel requires dense matrix, so we make one
        ColumnVector cv_lab = prhs(0).column_vector_value();
        Matrix m_dat = prhs(1).matrix_value();
				err = read_problem_dense(cv_lab, m_dat);
			}
			else {
        ColumnVector cv_lab = prhs(0).column_vector_value();
        SparseMatrix m_dat = prhs(1).sparse_matrix_value();
				err = read_problem_sparse(cv_lab, m_dat);
      }
		}
		else {
      ColumnVector cv_lab = prhs(0).column_vector_value();
      Matrix m_dat = prhs(1).matrix_value();
			err = read_problem_dense(cv_lab, m_dat);
    }

		// svmtrain's original code
		error_msg = svm_check_parameter(&prob, &param);

		if(err || error_msg)
		{
			if (error_msg != NULL)
				printf("Error: %s\n", error_msg);
			svm_destroy_param(&param);
			free(prob.y);
			free(prob.x);
			free(x_space);
			fake_answer(nlhs, plhs);
			return plhs;
		}

		if(cross_validation)
		{
			double ptr = do_cross_validation();
			plhs(0) = octave_value(ptr);
		}
		else
		{
			int nr_feat = (int)prhs(1).matrix_value().cols();
			const char *error_msg;
			model = svm_train(&prob, &param);
			error_msg = model_to_octave_structure(plhs, nr_feat, model);
			if(error_msg)
				printf("Error: can't convert libsvm model to matrix structure: %s\n", error_msg);
			svm_free_and_destroy_model(&model);
		}
		svm_destroy_param(&param);
		free(prob.y);
		free(prob.x);
		free(x_space);
    return plhs;
	}
	else
	{
		exit_with_help();
		fake_answer(nlhs, plhs);
		return plhs;
	}
}
