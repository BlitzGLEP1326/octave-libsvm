/**
 * Octave LIBSVM wrapper
 * This file contains the wrapper function for classifying samples given a model.
 * 
 * Adapted by Alan Meeson 2nd October 2014 based on the matlab version included with the LIBSVM library.
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

int print_null(const char *s,...) {}
int (*info)(const char *fmt,...) = &printf;

void read_sparse_instance(const SparseMatrix &prhs, int index, struct svm_node *x)
{
	int i, j, low, high;
	octave_idx_type *ir, *jc;
	double *samples;

	ir = prhs.mex_get_ir();
	jc = prhs.mex_get_jc();
	samples = (double*)prhs.mex_get_data();

	// each column is one instance
	j = 0;
	low = (int)jc[index], high = (int)jc[index+1];
	for(i=low;i<high;i++)
	{
		x[j].index = (int)ir[i] + 1;
		x[j].value = samples[i];
		j++;
	}
	x[j].index = -1;
}

static void fake_answer(int nlhs, octave_value_list &plhs)
{
	int i;
	for(i=0;i<nlhs;i++) plhs(i) = Matrix(0,0);
}

void predict(int nlhs, octave_value_list &plhs, const octave_value_list &prhs, struct svm_model *model, const int predict_probability)
{
	
	int label_vector_row_num, label_vector_col_num;
	int feature_number, testing_instance_number;
	int instance_index;
	double *ptr_instance, *ptr_label, *ptr_predict_label; 
	double *ptr_prob_estimates, *ptr_dec_values, *ptr;
	struct svm_node *x;
	SparseMatrix pplhs(0,0); // transposed instance sparse matrix
	octave_value_list tplhs(3); // temporary storage for plhs[]
	
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);
	double *prob_estimates=NULL;

	// prhs[1] = testing instance matrix
	feature_number = (int)prhs(1).columns();
	testing_instance_number = (int)prhs(1).rows();
	label_vector_row_num = (int)prhs(0).rows();
	label_vector_col_num = (int)prhs(0).columns();

	if(label_vector_row_num!=testing_instance_number)
	{
		printf("Length of label vector does not match # of instances.\n");
		fake_answer(nlhs, plhs);
		return;
	}
	if(label_vector_col_num!=1)
	{
		printf("label (1st argument) should be a vector (# of column is 1).\n");
		fake_answer(nlhs, plhs);
		return;
	}

	ptr_instance = (double *)prhs(1).mex_get_data();// mxGetPr(prhs[1]);	
	ptr_label    = (double*)prhs(0).mex_get_data();//mxGetPr(prhs[0]);
	
	// transpose instance matrix
	Matrix t_data(0,0);
	if(prhs(1).is_sparse_type())
	{
		if(model->param.kernel_type == PRECOMPUTED)
		{
			// precomputed kernel requires dense matrix, so we make one
			t_data = prhs(1).matrix_value();
		}
		else
		{
			//If it's a sparse matrix with a non PRECOMPUTED kernel, transpose it
			pplhs = prhs(1).sparse_matrix_value().transpose();
		}
	} else {
		t_data = prhs(1).matrix_value();
	}
	ptr_instance = (double *)t_data.mex_get_data();// mxGetPr(prhs[1]);
	if(predict_probability)
	{
		if(svm_type==NU_SVR || svm_type==EPSILON_SVR)
			info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
		else
			prob_estimates = (double *) malloc(nr_class*sizeof(double));
	}

	ColumnVector cv_predictions(testing_instance_number);
	tplhs(0) = cv_predictions;
	//tplhs[0] = mxCreateDoubleMatrix(testing_instance_number, 1, mxREAL);
	if(predict_probability)
	{
		// prob estimates are in plhs[2]
		if(svm_type==C_SVC || svm_type==NU_SVC) {
			Matrix m_pe(testing_instance_number, nr_class);
			tplhs(2) = m_pe;
			//tplhs[2] = mxCreateDoubleMatrix(testing_instance_number, nr_class, mxREAL);
		} else {
			Matrix m_pe(0,0);
			tplhs(2) = m_pe;
			//tplhs[2] = mxCreateDoubleMatrix(0, 0, mxREAL);
		}
	}
	else
	{
		// decision values are in plhs[2]
		if(svm_type == ONE_CLASS ||
		   svm_type == EPSILON_SVR ||
		   svm_type == NU_SVR ||
		   nr_class == 1) {// if only one class in training data, decision values are still returned.
			//tplhs[2] = mxCreateDoubleMatrix(testing_instance_number, 1, mxREAL);
			Matrix m_pe(testing_instance_number, 1);
			tplhs(2) = m_pe;
		} else {
			//tplhs[2] = mxCreateDoubleMatrix(testing_instance_number, nr_class*(nr_class-1)/2, mxREAL);
			Matrix m_pe(testing_instance_number, nr_class*(nr_class-1)/2);
			tplhs(2) = m_pe;
		}
	}

	ptr_predict_label = (double*)tplhs(0).column_vector_value().mex_get_data(); //mxGetPr(tplhs[0]);
	ptr_prob_estimates = (double*)tplhs(2).matrix_value().mex_get_data();//mxGetPr(tplhs[2]);
	ptr_dec_values = (double*)tplhs(2).matrix_value().mex_get_data();//mxGetPr(tplhs[2]);
	x = (struct svm_node*)malloc((feature_number+1)*sizeof(struct svm_node) );
	for(instance_index=0;instance_index<testing_instance_number;instance_index++)
	{
		int i;
		double target_label, predict_label;

		target_label = ptr_label[instance_index];

		if(prhs(1).is_sparse_type() && model->param.kernel_type != PRECOMPUTED) {// prhs[1]^T is still sparse
			read_sparse_instance(pplhs, instance_index, x);
		} else
		{
			for(i=0;i<feature_number;i++)
			{
				x[i].index = i+1;
				x[i].value = ptr_instance[testing_instance_number*i+instance_index];
			}
			x[feature_number].index = -1;
		}

		if(predict_probability)
		{
			if(svm_type==C_SVC || svm_type==NU_SVC)
			{
				predict_label = svm_predict_probability(model, x, prob_estimates);
				ptr_predict_label[instance_index] = predict_label;
				for(i=0;i<nr_class;i++)
					ptr_prob_estimates[instance_index + i * testing_instance_number] = prob_estimates[i];
			} else {
				predict_label = svm_predict(model,x);
				ptr_predict_label[instance_index] = predict_label;
			}
		}
		else
		{
			if(svm_type == ONE_CLASS ||
			   svm_type == EPSILON_SVR ||
			   svm_type == NU_SVR)
			{
				double res;
				predict_label = svm_predict_values(model, x, &res);
				ptr_dec_values[instance_index] = res;
			}
			else
			{
				double *dec_values = (double *) malloc(sizeof(double) * nr_class*(nr_class-1)/2);
				predict_label = svm_predict_values(model, x, dec_values);
				if(nr_class == 1) 
					ptr_dec_values[instance_index] = 1;
				else
					for(i=0;i<(nr_class*(nr_class-1))/2;i++)
						ptr_dec_values[instance_index + i * testing_instance_number] = dec_values[i];
				free(dec_values);
			}
			ptr_predict_label[instance_index] = predict_label;
		}

		if(predict_label == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}
	if(svm_type==NU_SVR || svm_type==EPSILON_SVR)
	{
		info("Mean squared error = %g (regression)\n",error/total);
		info("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
			((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
			);
	}
	else
		info("Accuracy = %g%% (%d/%d) (classification)\n",
			(double)correct/total*100,correct,total);

	// return accuracy, mean squared error, squared correlation coefficient
	ColumnVector cv_acc(3);
	
	//tplhs[1] = mxCreateDoubleMatrix(3, 1, mxREAL);
	ptr = (double*)cv_acc.mex_get_data();//mxGetPr(tplhs[1]);
	ptr[0] = (double)correct/total*100;
	ptr[1] = error/total;
	ptr[2] = ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
				((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt));
	tplhs(1) = cv_acc;
	free(x);
	if(prob_estimates != NULL)
		free(prob_estimates);

	switch(nlhs)
	{
		case 3:
			plhs(2) = tplhs(2);
			plhs(1) = tplhs(1);
		case 1:
		case 0:
			plhs(0) = tplhs(0);
	}
}

void exit_with_help()
{
	printf(
		"Usage: [predicted_label, accuracy, decision_values/prob_estimates] = svmpredict(testing_label_vector, testing_instance_matrix, model, 'libsvm_options')\n"
		"       [predicted_label] = svmpredict(testing_label_vector, testing_instance_matrix, model, 'libsvm_options')\n"
		"Parameters:\n"
		"  model: SVM model structure from svmtrain.\n"
		"  libsvm_options:\n"
		"    -b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet\n"
		"    -q : quiet mode (no outputs)\n"
		"Returns:\n"
		"  predicted_label: SVM prediction output vector.\n"
		"  accuracy: a vector with accuracy, mean squared error, squared correlation coefficient.\n"
		"  prob_estimates: If selected, probability estimate vector.\n"
	);
}

//void mexFunction( int nlhs, mxArray *plhs[],
//		 int nrhs, const mxArray *prhs[] )
//{
DEFUN_DLD (svmpredict, prhs, nargout,
           "[predicted_label, accuracy, decision_values/prob_estimates] = svmpredict(testing_label_vector, testing_instance_matrix, model, 'libsvm_options')")
{
	int nlhs = nargout;
	int nrhs = prhs.length();
	octave_value_list plhs(nlhs);
	int prob_estimate_flag = 0;
	struct svm_model *model;
	info = &print_null; //&mexPrintf;

	if(nlhs == 2 || nlhs > 3 || nrhs > 4 || nrhs < 3)
	{
		exit_with_help();
		fake_answer(nlhs, plhs);
		return plhs;
	}

	if(!prhs(0).is_double_type() || !prhs(1).is_double_type()) {
		printf("Error: label vector and instance matrix must be double\n");
		fake_answer(nlhs, plhs);
		return plhs;
	}

	if(prhs(2).is_map())
	{
		const char *error_msg;

		// parse options
		if(nrhs==4)
		{
			int i, argc = 1;
			char cmd[CMD_LEN], *argv[CMD_LEN/2];

			// put options in argv[]
			strncpy(cmd, prhs(3).string_value().c_str(), CMD_LEN);
			if((argv[argc] = strtok(cmd, " ")) != NULL)
				while((argv[++argc] = strtok(NULL, " ")) != NULL)
					;

			for(i=1;i<argc;i++)
			{
				if(argv[i][0] != '-') break;
				if((++i>=argc) && argv[i-1][1] != 'q')
				{
					exit_with_help();
					fake_answer(nlhs, plhs);
					return plhs;
				}
				switch(argv[i-1][1])
				{
					case 'b':
						prob_estimate_flag = atoi(argv[i]);
						break;
					case 'q':
						i--;
						info = &print_null;
						break;
					default:
						printf("Unknown option: -%c\n", argv[i-1][1]);
						exit_with_help();
						fake_answer(nlhs, plhs);
						return plhs;
				}
			}
		}
		octave_scalar_map osm_model = prhs(2).scalar_map_value();
		model = octave_matrix_to_model(osm_model, &error_msg);
		if (model == NULL)
		{
			printf("Error: can't read model: %s\n", error_msg);
			fake_answer(nlhs, plhs);
			return plhs;
		}

		if(prob_estimate_flag)
		{
			if(svm_check_probability_model(model)==0)
			{
				printf("Model does not support probabiliy estimates\n");
				fake_answer(nlhs, plhs);
				svm_free_and_destroy_model(&model);
				return plhs;
			}
		}
		else
		{
			if(svm_check_probability_model(model)!=0)
				info("Model supports probability estimates, but disabled in predicton.\n");
		}

		predict(nlhs, plhs, prhs, model, prob_estimate_flag);
		// destroy model
		svm_free_and_destroy_model(&model);
	}
	else
	{
		printf("model file should be a struct array\n");
		fake_answer(nlhs, plhs);
	}

	return plhs;
}
