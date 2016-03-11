/*
Fast Artificial Neural Network Library (fann)
Copyright (C) 2003-2016 Steffen Nissen (steffen.fann@gmail.com)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <stdio.h>

#include "fann.h"
#include "parallel_fann.h"

float get_MSE(struct fann *ann, struct fann_train_data *data) {
	int i;
	fann_reset_MSE(ann);
	for(i = 0; i < fann_length_train_data(data); i++)
	{
		fann_test(ann, data->input[i], data->output[i]);
	}
	
	return fann_get_MSE(ann);
}

int main()
{
	const unsigned int num_layers = 5;
	const unsigned int num_neurons_hidden = 96;
	const float desired_error = (const float) 0.004;
	const unsigned int max_epochs = 6000;
	float error, error_dropout, error_test, error_test_dropout;
	struct fann *ann, *ann_dropout;
	struct fann_train_data *train_data, *test_data;

	unsigned int i = 0;

	printf("Creating network.\n");

	train_data = fann_read_train_from_file("../datasets/robot.train");
		test_data = fann_read_train_from_file("../datasets/robot.test");

	ann = fann_create_standard(num_layers,
					  train_data->num_input, num_neurons_hidden, num_neurons_hidden, num_neurons_hidden, train_data->num_output);

	printf("Training network.\n");

	fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL);
	fann_set_learning_momentum(ann, 0.4f);

	ann_dropout = fann_copy(ann);

	fann_set_do_dropout(ann_dropout, 1);
	fann_set_dropout_fraction(ann_dropout, 0.15f);
	
	/* seed the random number generator when using dropout */
	fann_seed_rand();

	for(i = 1; i <= max_epochs; i++)
	{
		error = fann_train_epoch_parallel(ann, train_data,8);
		error_dropout = fann_train_epoch_parallel(ann_dropout, train_data,8);
		error_test = get_MSE(ann, test_data);
		error_test_dropout = get_MSE(ann_dropout, test_data);
		printf("Epochs     %8d. TRAIN ERROR dropout: %.10f - no dropout: %.10f    TEST ERROR dropout: %.10f - no dropout: %.10f\n", i, error_dropout, error, error_test_dropout, error_test);

		if ((error_test <= desired_error) || (error_test_dropout <= desired_error)) {
			break;
		}
	}

	printf("Testing network.\n");
	
	printf("MSE error on test data without dropout: %f\n", get_MSE(ann, test_data));

	printf("Saving network.\n");

	fann_save(ann, "robot_float.net");

	printf("MSE error on test data with dropout: %f\n", get_MSE(ann_dropout, test_data));

	printf("Saving network.\n");

	fann_save(ann_dropout, "robot_float_dropout.net");

	printf("Cleaning up.\n");
	fann_destroy_train(train_data);
	fann_destroy_train(test_data);
	fann_destroy(ann);
	fann_destroy(ann_dropout);


	return 0;
}
