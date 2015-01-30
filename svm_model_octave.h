/**
 * Octave LIBSVM wrapper
 * This file contains declarations of functions for converting between 
 * an Octave structure and LIBSVM's model format.
 * Created by Alan Meeson 2nd October 2014
 */

#include <octave/oct.h>
#include <octave/ov-struct.h>
#include "../svm.h"
const char *model_to_octave_structure(octave_value_list &plhs, int num_of_feature, struct svm_model *model);
struct svm_model *octave_matrix_to_model(octave_scalar_map &octave_struct, const char **error_message);
